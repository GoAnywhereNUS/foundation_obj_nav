import math
import torch
import numpy as np
from utils.mapper import Mapper
from utils.fmm_planner import FMMPlanner
from utils.habitat_utils import ObjNavEnv, setup_env_config

import habitat
import skimage
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import home_robot.utils.pose as pu
from home_robot.core.interfaces import (
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)
from home_robot.utils.geometry import (
    xyt_global_to_base
)
from home_robot.utils.depth import (
    get_camera_matrix,
    get_point_cloud_from_z_t,
    transform_camera_view_t
)
import cv2

class Controller:
    def __init__(
        self, 
        device=torch.device('cuda:0')
    ):
        self.curr_goal = None
        self.device = device

    def set_subgoal_coord(self, subgoal, obs=None):
        """
        Set the controller's current subgoal as coordinates in world frame. 
        If needed, the controller can take in current sensor observations 
        to ground the subgoal.
        """
        raise NotImplementedError
    
    def set_subgoal_image(self, subgoal, obs):
        """
        Set the controller's current subgoal as an image crop.
        """
        raise NotImplementedError
    
    def reset_subgoal(self):
        """
        Clears the previously set subgoal and resets state.
        """
        raise NotImplementedError
    
    def step(self):
        """
        Plans a path toward the set goal, and takes a step towards it.
        """
        raise NotImplementedError
    
    def update(self, obs):
        """
        Update the internal map from observations.
        """
        raise NotImplementedError
    
    def visualise(self, obs):
        """
        Visualise the observations, internal map, pose (and plan?)

        Input:
            obs: home_robot Observation type

        Output:
            image: cv2 Image combining all the above visuals
        """
        raise NotImplementedError
    

class FMMController(Controller):
    def __init__(
        self,
        device,
        controller_config=None,
        mapper_config=None,
    ):
        super().__init__(device)

        self.mapper = Mapper(device)
        self.planner = None
        self.discrete_actions = True
        self.agent_cell_radius = 4
        self.map_resolution = 5
        self.turn_angle = 30
        self.step_size = 5
        self.scale = 1.0
        self.visualise_planner = True
        self.visualise_subgoal_selection = False
        self.vis_dir="logs/planning"
        self.goal_tolerance = np.sqrt(0.25**2 * 2) # ~0.36m
        self.max_goal_search_dist = 1.0
        self.select_subgoal_downsample_depth = 1 # int

        self.obs_dilation_selem_radius = 2
        self.obs_dilation_selem = skimage.morphology.disk(
            self.obs_dilation_selem_radius
        )

        # Goal setting
        self.set_goal_global = None
        self.curr_tracking_goal_global = None
        self.set_goal_pose_global = None

        # Debug
        self.vis_dist_map = None
        self.goal_local_cell = None
        self.subgoal_local_cell = None
        self.set_goal_pose_local_cell = None

    def set_subgoal_coord(self, subgoal, obs):
        """
        Set the subgoal for the controller to plan and track towards.

        Input:
            subgoal: tuple, x-y coordinates relative to egocentric frame 
                     (+X forward, +Y left, +Theta counterclockwise)
            obs: observation from the simulator environment
        """
        ego_x, ego_y = subgoal
        px, py, ptheta = obs['gps'][0], -obs['gps'][1], obs['compass'][0]
        T = torch.tensor([
            [np.cos(ptheta), -np.sin(ptheta),    px,],
            [np.sin(ptheta), np.cos(ptheta),     py,],
            [0.            , 0.,                 1.,],
        ], device=self.device)
        env_xy = torch.matmul(
            T, torch.tensor([ego_x, ego_y, 1.], device=self.device)
        )[:2]
        self.set_goal_global = self.mapper.env_to_global_map_pose(
            torch.cat((env_xy, torch.tensor([0.], device=self.device)))
        )[:2]
        self.curr_tracking_goal_global = self.set_goal_global
        self.set_goal_pose_global = self.mapper.env_to_global_map_pose(
            torch.tensor([px, py, ptheta], device=self.device)
        )
        print("Set goal as:", self.set_goal_global)

    def set_subgoal_image(self, subgoal, cam_uuid, obs, camera_matrix):
        """
        Set the subgoal for the controller to plan and track towards.

        Input:
            subgoal: tuple, bounding box of ROI specifying the subgoal in obs, i.e. (min_x, max_x, min_y, max_y)
            obs: observation from the simulator environment
        """
        min_x, min_y, max_x, max_y = subgoal
        depth = torch.tensor(
            np.transpose(obs[cam_uuid], axes=[2, 0, 1]), 
            device=self.device
        )

        # Get point cloud from depth map
        point_cloud_t = get_point_cloud_from_z_t(
            depth, camera_matrix, self.device, 
            scale=self.select_subgoal_downsample_depth
        )
        point_cloud_base_coords = transform_camera_view_t(
            point_cloud_t, 0.88, np.zeros(1), self.device
        )
        points = point_cloud_base_coords[0][min_y:max_y, min_x:max_x].reshape(-1, 3)

        # Change axes to +X forward, +Y left and +Z up
        points = points[:, [1, 0, 2]] * torch.tensor([1., -1., 1.], device=self.device)

        if self.visualise_subgoal_selection:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            scatter_points = points.cpu().numpy()
            ax.scatter(
                scatter_points[:, 0], 
                scatter_points[:, 1], 
                scatter_points[:, 2],
                c=scatter_points[:, 2],
                cmap=plt.cm.viridis
            )
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.savefig('logs/planning/pointcloud.png')
            plt.clf()

            plt.imshow(obs[cam_uuid].squeeze()[min_y:max_y, min_x:max_x])
            plt.savefig("logs/planning/depth_crop_" + str(cam_uuid) + ".png")
            plt.clf()

        min_x, max_x = torch.min(points[:, 0]).item(), torch.max(points[:, 0]).item()
        min_y, max_y = torch.min(points[:, 1]).item(), torch.max(points[:, 1]).item()
        min_h, max_h = torch.min(points[:, 2]).item(), torch.max(points[:, 2]).item()

        subgoal = [
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
        ]

        print("### Select image subgoal ###")
        print("Bounding box:", min_x, max_x, min_y, max_y, min_h, max_h)
        print("Image subgoal:", subgoal)

        self.set_subgoal_coord(subgoal, obs)

    def reset_subgoal(self):
        self.set_goal_global = None
        self.curr_tracking_goal_global = None
        self.set_goal_pose_global = None
    
    def update(self, obs):
        self.mapper.update(obs)

    def visualise(self, obs):
        vis_image = self.mapper.visualise(obs)

        # Debugging. Add in distance field visualisation.
        if self.vis_dist_map is not None:
            dist_map_vis = Image.new("P", self.vis_dist_map.shape)
            dist_map_vis.putdata(self.vis_dist_map.flatten().astype(np.uint8))
            dist_map_vis = dist_map_vis.convert("RGB")
            dist_map_vis = cv2.resize(
                np.array(dist_map_vis), (480, 480), interpolation=cv2.INTER_NEAREST
            )
            dist_map_vis[
                self.goal_local_cell[0] - 1 : self.goal_local_cell[0] + 2, 
                self.goal_local_cell[1] - 1 : self.goal_local_cell[1] + 2,
                2
            ] = 255
            dist_map_vis[
                self.subgoal_local_cell[0] - 1 : self.subgoal_local_cell[0] + 2,
                self.subgoal_local_cell[1] - 1 : self.subgoal_local_cell[1] + 2,
                1:
            ] = np.array([255, 255])
            dist_map_vis[dist_map_vis.shape[0] // 2, dist_map_vis.shape[1] // 2, 1] = 255
            dist_map_vis[
                self.set_goal_pose_local_cell[0] - 1 : self.set_goal_pose_local_cell[0] + 2,
                self.set_goal_pose_local_cell[1] - 1 : self.set_goal_pose_local_cell[1] + 2,
                :2
            ] = np.array([255, 255])

            vis_image[50:530, 670:1310] = np.ones((480, 640, 3)) * 255
            dist_map_vis = np.flipud(dist_map_vis)
            vis_image[50:530, 750:1230] = dist_map_vis

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (20, 255, 20)  # BGR
            thickness = 2
            text = str(self.goal_local_cell[0]) + ", " + str(self.goal_local_cell[1])
            textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
            textX = 990
            textY = 600 + (textsize[1] // 2)
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

        return vis_image

    def step(self, update=True):
        if self.set_goal_global is None:
            return None, False

        obstacle_map = self.mapper.map_state.get_obstacle_map(0)
        
        # Get current pose and long-term goal
        px, py, po, ly1, _, lx1, _ = self.mapper.map_state.get_planner_pose_inputs(0)
        local_xy = self.mapper.global_pose_to_local_map_cell(
            torch.tensor([px, py, po], device=self.device)
        )[:2]
        curr_grid_pose = [int(c) for c in local_xy]

        # Dilate obstacles to get traversability map
        obstacle_map = np.rint(obstacle_map)
        dilated_obstacles = cv2.dilate(obstacle_map, self.obs_dilation_selem, iterations=1)
        traversable = 1 - dilated_obstacles
        agent_rad = self.agent_cell_radius
        traversable[
            curr_grid_pose[0] - agent_rad : curr_grid_pose[0] + agent_rad + 1,
            curr_grid_pose[1] - agent_rad : curr_grid_pose[1] + agent_rad + 1,
        ] = 1
        traversable = self._add_boundary(traversable)

        if self.planner is None or update:
            self.planner = FMMPlanner(
                traversable,
                step_size=self.step_size,
                vis_dir=self.vis_dir,
                visualize=False,
                print_images=self.visualise_planner,
                goal_tolerance=self.goal_tolerance,
            )

        curr_goal_local_cell = self.mapper.global_pose_to_local_map_cell(
            torch.cat([
                self.curr_tracking_goal_global, torch.tensor([1.], device=self.device)
            ])
        )
        init_goal_local_cell = self.mapper.global_pose_to_local_map_cell(
            torch.cat([
                self.set_goal_global, torch.tensor([1.], device=self.device)
            ])
        )
        set_goal_pose_cell = self.mapper.global_pose_to_local_map_cell(
            torch.cat([
                self.set_goal_pose_global, torch.tensor([1.], device=self.device)
            ])
        )

        self.set_goal_pose_local_cell = set_goal_pose_cell
        success, updated_goal = self.planner.set_goal(
            curr_goal_local_cell,
            auto_improve=True,
            init_goal=init_goal_local_cell,
            pose=set_goal_pose_cell,
            max_dist_in_cells=int(
                self.max_goal_search_dist * 100.0 / self.map_resolution
            )
        )

        if not success:
            print("Failed to set goal!")
            # TODO: Handle this failure gracefully
            raise NotImplementedError
        
        if updated_goal is not None:
            updated_goal_pose = self.mapper.local_map_cell_to_global_pose(updated_goal)
            print(
                "Initial goal: ", 
                self.set_goal_pose_global, 
                "   Updating from:", 
                self.curr_tracking_goal_global, 
                " -> ", 
                updated_goal_pose
            )
            self.curr_tracking_goal_global = updated_goal_pose

        self.vis_dist_map = self.planner.fmm_dist
        self.goal_local_cell = curr_goal_local_cell
        stg_x, stg_y, replan, stop = self.planner.get_short_term_goal(
            curr_grid_pose,
            visualize=False
        )

        if replan:
            # TODO: Handle replanning if too near an obstacle
            print("Replanning...")

        if stop:
            print("Reached goal!")
            # Reset goal
            self.reset_subgoal()
            
            # Reset debug stuff
            self.vis_dist_map = None
            self.goal_local_cell = None
            self.subgoal_local_cell = None

            return None, True

        ego_stg_x, ego_stg_y = [
            int(stg_x) - curr_grid_pose[0],
            int(stg_y) - curr_grid_pose[1],
        ]
        curr_pose_angle = pu.normalize_angle(po)
        angle_to_goal = math.degrees(math.atan2(ego_stg_x, ego_stg_y))
        ego_stg_angle = pu.normalize_angle(curr_pose_angle - angle_to_goal)

        action = self._get_servo_action(
            [ego_stg_x, ego_stg_y, ego_stg_angle],
            [px, py, curr_pose_angle],
        )
        # action = None

        self.subgoal_local_cell = [int(stg_x), int(stg_y)]
        # print(">>>>")
        # print("Replan:", replan, "  Stop:", stop)
        # print("****")

        return action, False

    def _get_servo_action(self, rel_stg, curr_pose):
        stg_x, stg_y, stg_angle = rel_stg
        px, py, po = curr_pose

        if self.discrete_actions:
            if stg_angle > self.turn_angle / 2.0:
                # action = DiscreteNavigationAction.TURN_RIGHT
                action = "turn_right"
            elif stg_angle < -self.turn_angle / 2.0:
                # action = DiscreteNavigationAction.TURN_LEFT
                action = "turn_left"
            else:
                # action = DiscreteNavigationAction.MOVE_FORWARD
                action = "move_forward"
        else:
            # Use the short-term goal to set where we should be heading next
            # Must return commands in radians and meters
            scale = 0.01 * self.map_resolution
            m_stg_x, m_stg_y = scale * stg_x, scale * stg_y
            rad_stg_angle = stg_angle * np.pi / 180.0

            if np.abs(rad_stg_angle) > self.turn_angle / 2.0:
                action = ContinuousNavigationAction([0, 0, -rad_stg_angle])
            else:
                # relative_angle_goal = math.radians(relative_angle_goal)
                # action = ContinuousNavigationAction([m_relative_stg_y, m_relative_stg_x, -relative_angle])

                # xyt_global = [m_stg_y, m_stg_x, -rad_stg_angle]
                xyt_global = [m_stg_x, m_stg_y, -rad_stg_angle]

                xyt_local = xyt_global_to_base(
                    xyt_global, [0, 0, po * np.pi / 180.0]
                )
                xyt_local[
                    2
                ] = -rad_stg_angle  # the original angle was already in base frame
                action = ContinuousNavigationAction(xyt_local)

        return action

    def _add_boundary(self, mat: np.ndarray, value=1) -> np.ndarray:
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2)) + value
        new_mat[1 : h + 1, 1 : w + 1] = mat
        return new_mat
    
    def _remove_boundary(self, mat: np.ndarray, value=1) -> np.ndarray:
        return mat[value:-value, value:-value]


if __name__ == "__main__":
    device = torch.device('cuda:0')
    controller = FMMController(device)

    config = setup_env_config()
    env = ObjNavEnv(habitat.Env(config=config), config)
    obs = env.reset()
    import cv2
    import time
    cv2.namedWindow("Images")
    auto = False

    while True:
        obs = env.get_observation()

        # Map
        t1 = time.time()
        controller.update(obs)
        print("Mapping:", time.time() - t1, "   Pose:", obs['gps'][0], obs['gps'][1], math.degrees(obs['compass']))

        # Visualise
        images = controller.visualise(obs)
        cv2.imshow("Images", images)
        key = cv2.waitKey(50)

        action = None
        if key == ord('w'):
            action = 'move_forward'
        elif key == ord('a'):
            action = 'turn_left'
        elif key == ord('d'):
            action = 'turn_right'
        elif key == ord('q'):
            import sys; sys.exit(0)
        elif key == ord('k'):
            auto = True
        elif key == ord('l'):
            auto = False
        elif key == ord('p'):
            controller.set_subgoal_coord([0.5, 0], obs)
        elif key == ord('o'):
            controller.set_subgoal_image(
                [260, 380, 120, 360], 
                "forward_depth",
                obs,
                get_camera_matrix(640, 480, 90)
            )

        if auto:
            action, stop = controller.step()
            # _, stop = controller.step()
            if stop:
                auto = False
            print("(Auto) Action:", action)

        if action is not None:
            env.act(action)