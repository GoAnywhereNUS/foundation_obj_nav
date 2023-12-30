import cv2
import math
import torch
import numpy as np
from PIL import Image

import home_robot.utils.pose as pu
import home_robot.utils.visualization as vu
from home_robot.mapping.geometric.geometric_map_module import GeometricMapModule
from home_robot.mapping.geometric.geometric_map_state import GeometricMapState
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import(
    Categorical2DSemanticMapModule,
)
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.constants import MapConstants as MC


class Mapper:
    def __init__(
        self, 
        device, 
        semantic_categories=None,
        semantic_annotations=None,
        traversable_categories=None,
    ):
        self.device = device
        self.semantic = semantic_categories is not None
        self.semantic_categories = semantic_categories
        if self.semantic:
            assert semantic_annotations is not None
        self.semantic_annotations = semantic_annotations
        self.traversable_categories = (
            None
            if isinstance(traversable_categories, list) and len(traversable_categories) == 0
            else traversable_categories
        )

        # Assume that semantic categories will always start
        # with a catch-all, i.e. ["others", ...]
        if self.semantic:
            one_hot_encoding = torch.eye(len(semantic_categories))
            self.instance_to_one_hot = torch.stack([
                (
                    one_hot_encoding[semantic_categories.index(cat)]
                    if cat in semantic_categories
                    else one_hot_encoding[0]
                ) for cat in semantic_annotations
            ], dim=0)

        if self.semantic:
            self.map_state = Categorical2DSemanticMapState(
                device=device,
                num_environments=1,
                num_sem_categories=len(self.semantic_categories),
                map_resolution=5,
                map_size_cm=4800,
                global_downscaling=2,
            )
            self._mapper = Categorical2DSemanticMapModule(
                frame_height=480,            # pix
                frame_width=640,            # pix
                camera_height=0.88,         # m
                hfov=90,                    # deg
                num_sem_categories=len(semantic_categories),
                map_size_cm=4800,           # cm
                map_resolution=5,           # cm
                vision_range=100,           # no. of map cells
                explored_radius=150,        # cm
                been_close_to_radius=200,   # cm
                global_downscaling=2,
                du_scale=4,
                cat_pred_threshold=1.0,
                exp_pred_threshold=1.0,
                map_pred_threshold=1.0,
                min_obs_height_cm=50,
                min_depth=0.5,              # m
                max_depth=5.0,              # m
            )
        else:
            self.map_state = GeometricMapState(
                device=device,
                num_environments=1,
                map_resolution=5,
                map_size_cm=4800,
                global_downscaling=2,
            )
            self._mapper = GeometricMapModule(
                frame_height=480,           # pix
                frame_width=640,            # pix
                camera_height=0.88,         # m
                hfov=90,                    # deg
                map_size_cm=4800,           # cm
                map_resolution=5,           # cm
                vision_range=100,           # no. of map cells
                explored_radius=150,        # cm
                been_close_to_radius=200,   # cm
                global_downscaling=2,
                du_scale=4,
                exp_pred_threshold=1.0,
                map_pred_threshold=1.0,
                min_obs_height_cm=50,
                min_depth=0.5,              # m
                max_depth=5.0,              # m
            )

        self.mapper = torch.nn.DataParallel(self._mapper, device_ids = [0,2,3])
        self.last_pose = np.zeros(3)
        self.init = False
        self.T_env_flipped_to_global = None
        self.T_global_to_env_flipped = None
        self.init_angle_env_frame = None

    def _preprocess(self, obs):
        rgb_tensor = torch.from_numpy(obs['forward_rgb'])
        depth_tensor = torch.from_numpy(obs['forward_depth']) * 100.0 # convert m -> cm

        if self.semantic:
            semantic_obs = torch.from_numpy(obs['forward_semantic'])[:, :, 0]
            semantic_tensor = self.instance_to_one_hot[semantic_obs]
            seq_obs = torch.cat(
                (rgb_tensor, depth_tensor, semantic_tensor), dim=-1
            ).permute(2, 0, 1).to(self.device)
        else:
            seq_obs = torch.cat(
                (rgb_tensor, depth_tensor), dim=-1
            ).permute(2, 0, 1).to(self.device)

        curr_pose = np.array([obs['gps'][0], -obs['gps'][1], obs['compass'][0]])
        pose_delta = torch.Tensor(
            pu.get_rel_pose_change(curr_pose, self.last_pose)
        ).unsqueeze(0)
        self.last_pose = curr_pose

        camera_pose = None

        return (
            seq_obs,
            pose_delta,
            camera_pose
        )
    
    def _set_pose_offset(self, obs):
        # Assumes that map global pose has just been reinitialised.
        env_pose_flipped_y = [
            obs['gps'][0], -obs['gps'][1], math.degrees(obs['compass'])
        ]

        delta_x = self.map_state.global_pose[0, 0] - env_pose_flipped_y[0]
        delta_y = self.map_state.global_pose[0, 1] - env_pose_flipped_y[1]

        # Assume that map global pose is always initialised to 0 deg.
        # I.e. current pose angle is angle of map global frame relative to env frame.
        delta_o = pu.normalize_angle(math.degrees(-env_pose_flipped_y[2]))
        self.T_env_flipped_to_global = torch.tensor([
            [np.cos(delta_o),   -np.sin(delta_o),   delta_x],
            [np.sin(delta_o),   np.cos(delta_o),    delta_y],
            [0.,                0.,                 1.],
        ], device=self.device).float()

        # print(">>>", delta_x, delta_y, delta_o)
        # print(self.T_env_flipped_to_global)

        self.T_global_to_env_flipped = torch.eye(3, device=self.device)
        self.T_global_to_env_flipped[:2, :2] = self.T_env_flipped_to_global[:2, :2].T
        self.T_global_to_env_flipped[:2, 2] = -torch.matmul(
            self.T_env_flipped_to_global[:2, :2].T,
            self.T_env_flipped_to_global[:2, 2]
        ).float()

        self.init_angle_env_frame = env_pose_flipped_y[2]
    
    def env_to_global_map_pose(self, env_pose):
        pos_h = torch.cat([env_pose[:2], torch.tensor([1.], device=self.device)])
        g_xy = torch.matmul(self.T_env_flipped_to_global, pos_h)[:2]
        go = pu.normalize_angle(env_pose[2].item() - self.init_angle_env_frame)
        return torch.cat([g_xy, torch.tensor([go], device=self.device)])

    def global_map_to_env_pose(self, global_pose):
        pos_h = torch.cat([global_pose[:2], torch.tensor([1.], device=self.device)])
        e_xy = torch.matmul(self.T_global_to_env_flipped, pos_h)[:2]
        eo = pu.normalize_angle(global_pose[2].item() + self.init_angle_env_frame)
        return torch.cat([e_xy, torch.tensor([eo], device=self.device)])

    def local_map_cell_to_global_pose(self, local_map_cell):
        """
        Converts from a global pose to local map cell.

        Inputs:
            local_map_cell: List, [x_coords, y_coords]

        Outputs:
            global_pose: Simply initialises the global pose orientation as 0.
        """
        lx, ly = local_map_cell
        global_pose = torch.tensor([
            (lx + 0.5) * self.map_state.resolution / 100.0,
            (ly + 0.5) * self.map_state.resolution / 100.0,
            0.
        ], device=self.device)
        global_pose[:2] += self.map_state.origins[0, :2]
        return global_pose

    def global_pose_to_local_map_cell(self, global_pose):
        """
        Converts from a global pose to local map cell coordinates.
        """
        l_xy = global_pose[:2] - self.map_state.origins[0, :2]
        lx, ly = (
            int(l_xy[0].item() * 100.0 / self.map_state.resolution),
            int(l_xy[1].item() * 100.0 / self.map_state.resolution),
        )
        lx, ly = pu.threshold_poses(
            [lx, ly], self.map_state.local_map.shape[2:]
        )
        return [ly, lx, global_pose[2].item()]

    def reset(self):
        self.init = False
        self.T_env_flipped_to_global = None
        self.T_global_to_env_flipped = None
        self.init_angle_env_frame = None

    def update(self, obs):
        # Preprocess the observations from the simulator
        (
            seq_obs,
            pose_delta,
            camera_pose
        ) = self._preprocess(obs)

        # Update the map
        dones = torch.tensor([not self.init])
        update_globals = torch.tensor([True])

        (
            seq_map_feats,
            self.map_state.local_map,
            self.map_state.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.mapper(
            seq_obs.unsqueeze(0).unsqueeze(0),
            pose_delta.unsqueeze(0),
            dones.unsqueeze(0),
            update_globals.unsqueeze(0),
            camera_pose,
            self.map_state.local_map,
            self.map_state.global_map,
            self.map_state.local_pose,
            self.map_state.global_pose,
            self.map_state.lmb,
            self.map_state.origins
        )

        self.map_state.local_pose = seq_local_pose[:, -1]
        self.map_state.global_pose = seq_global_pose[:, -1]
        self.map_state.lmb = seq_lmb[:, -1]
        self.map_state.origins = seq_origins[:, -1]

        if not self.init:
            self._set_pose_offset(obs)

        self.init = True

    def get_obstacle_map(self):
        if self.semantic and self.traversable_categories is not None:
            obstacle_map = self.map_state.get_obstacle_map(0)
            for cat in self.traversable_categories:
                cat_idx = self.semantic_categories.index(cat)
                cat_map = self.map_state.local_map[0, MC.NON_SEM_CHANNELS + cat_idx].cpu().numpy()
                cat_mask = ~(cat_map > 0)
                obstacle_map *= cat_mask.astype(np.float)
            return obstacle_map
        else:
            return self.map_state.get_obstacle_map(0)


    def visualise(self, obs, semantic_category=None):
        rgb_frame = obs['forward_rgb']
        depth_frame = obs['forward_depth'][:, :, 0]
        if depth_frame.max() > 0:
            depth_frame = depth_frame / depth_frame.max()
        depth_frame = (depth_frame * 255).astype(np.uint8)
        depth_frame = np.repeat(depth_frame[:, :, np.newaxis], 3, axis=2)

        vis_image = np.ones((655, 1820, 3)).astype(np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (20, 20, 20)  # BGR
        thickness = 2

        text = "RGB"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (640 - textsize[0]) // 2 + 15
        textY = (50 + textsize[1]) // 2
        vis_image = cv2.putText(
            vis_image,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        text = "Depth"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = 640 + (640 - textsize[0]) // 2 + 30
        textY = (50 + textsize[1]) // 2
        vis_image = cv2.putText(
            vis_image,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        text = "Map"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = 1280 + (480 - textsize[0]) // 2 + 45
        textY = (50 + textsize[1]) // 2
        vis_image = cv2.putText(
            vis_image,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        map_color_palette = [
            1.0,
            1.0,
            1.0,  # empty space
            0.6,
            0.6,
            0.6,  # obstacles
            0.95,
            0.95,
            0.95,  # explored area
            0.96,
            0.36,
            0.26,  # visited area
            0.98,
            0.50,
            0.46,  # 1st semantic category
        ]
        map_color_palette = [int(x * 255.0) for x in map_color_palette]

        obstacle_map = self.map_state.get_obstacle_map(0)
        explored_map = self.map_state.get_explored_map(0)
        visited_map = self.map_state.get_visited_map(0)

        vis_map = np.zeros(obstacle_map.shape)
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        visited_mask = visited_map == 1
        vis_map[explored_mask] = 2
        vis_map[obstacle_mask] = 1
        vis_map[visited_mask] = 3

        # TODO: Extracting single category only from map_state. Figure 
        # out how to use the semantic map from map_state.get_semantic_map
        # instead for a more general approach.
        if semantic_category and type(semantic_category) == str:
            semantic_map = self.map_state.local_map[0, MC.NON_SEM_CHANNELS + 1].cpu().numpy()
            semantic_mask = semantic_map > 0
            vis_map[semantic_mask] = 4

        geometric_map_vis = Image.new("P", vis_map.shape)
        geometric_map_vis.putpalette(map_color_palette)
        geometric_map_vis.putdata(vis_map.flatten().astype(np.uint8))
        geometric_map_vis = geometric_map_vis.convert("RGB")
        geometric_map_vis = np.flipud(geometric_map_vis)
        geometric_map_vis = cv2.resize(
            geometric_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )
        vis_image[50:530, 1325:1805] = geometric_map_vis

        # Draw RGB
        vis_image[50:530, 15:655] = cv2.resize(rgb_frame, (640, 480))

        # Draw depth frame
        vis_image[50:530, 670:1310] = cv2.resize(depth_frame, (640, 480))
        
        # Draw agent arrow
        curr_x, curr_y, curr_o, gy1, _, gx1, _ = self.map_state.get_planner_pose_inputs(0)
        pos = (
            (curr_x * 100.0 / self.map_state.resolution - gx1)
            * 480
            / self.map_state.local_map_size,
            (self.map_state.local_map_size - curr_y * 100.0 / self.map_state.resolution + gy1)
            * 480
            / self.map_state.local_map_size,
            np.deg2rad(-curr_o),
        )
        agent_arrow = vu.get_contour_points(pos, origin=(1325, 50), size=10)
        color = map_color_palette[9:12]
        cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)

        return vis_image
