import math
import torch
import numpy as np
from enum import Enum
from utils.mapper import Mapper
from utils.fmm_planner import FMMPlanner
from utils.habitat_utils import ObjNavEnv, setup_env_config, sensor_config_dict

import habitat
import skimage
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import home_robot.utils.pose as pu
import home_robot.utils.rotation as ru
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
    transform_camera_view_t,
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
    
    def reset(self):
        """
        Clears controller state
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

class DiscreteRecovery:
    def __init__(self):
        self.prev_map_pose = None
        self.current_mode = "turn"
        self.max_num_turn_steps = 35

        # State tracking
        self.perturbation_actions = ["move_forward", "move_forward"]
        self.num_turn_steps = 0

    def get_recovery_action(self, pose):
        if self.prev_map_pose is None:
            self.prev_map_pose = pose
            self.num_turn_steps += 1
            action = "turn_left"
            done = False

            return action, done, self.current_mode

        px, py, ptheta = pose
        prev_x, prev_y, prev_theta = self.prev_map_pose

        # Finite state machine for recovery. Basically:
        # turn -> probe -> perturb -> done
        #  |  ^____|  |      ^__|
        #  V          V
        # failed     failed
        if self.current_mode == "turn":
            # Check effects of turn action (Note: currently ignore effects
            # because we assume turn always succeeds. Directly switch to probe.)
            self.current_mode = (
                "probe" if self.num_turn_steps < self.max_num_turn_steps
                else "failed"
            )
        
        elif self.current_mode == "probe":
            # Check effects of probing
            probe_success = abs(prev_x - px) > 0.05 or abs(prev_y - py) > 0.05
            self.current_mode = "perturb" if probe_success else "turn"

        elif self.current_mode == "perturb":
            self.current_mode = "perturb" if len(self.perturbation_actions) > 0 else "done"

        elif self.current_mode == "done":
            pass

        elif self.current_mode == "failed":
            pass

        else:
            raise NotImplementedError
        
        # Update flags, action to take based on current mode
        if self.current_mode == "turn":
            print(">>> Recovery: TURN_LEFT")
            action, done = "turn_left", False
            self.num_turn_steps += 1
        elif self.current_mode == "probe":
            print(">>> Recovery: PROBE FORWARD")
            action, done = "move_forward", False
        elif self.current_mode == "perturb":
            action, done = self.perturbation_actions[0], False
            print(">>> Recovery: PERTURB", action)
            self.perturbation_actions.pop(0)
        elif self.current_mode == "done" or self.current_mode == "failed":
            print(">>> Recovery: TERMINATING (", self.current_mode, ")")
            action, done = None, True

        return action, done, self.current_mode

class FMMController(Controller):
    def __init__(
        self,
        device,
        env_config=None,
        controller_config=None,
        mapper_config=None,
    ):
        super().__init__(device)

        if env_config is not None:
            self.turn_angle = env_config.habitat.simulator.turn_angle
        else:
            self.turn_angle = 30    # degrees

        self.mapper = Mapper(device)
        self.planner = None
        self.discrete_actions = True
        self.agent_cell_radius = 4
        self.map_resolution = 5
        self.turn_angle = 30
        self.step_size = 5
        self.scale = 1.0
        self.visualise_planner = True
        self.visualise_subgoal_selection = True
        self.vis_dir="logs/planning"
        self.init_goal_tolerance = 2.0  # tolerance of initially declared goal
        self.curr_goal_tolerance = 0.5  # tolerance of currently tracking goal
        self.max_goal_search_dist = 1.0
        self.select_subgoal_downsample_depth = 1 # int
        self.perform_recovery = False

        self.obs_dilation_selem_radius = 2
        self.obs_dilation_selem = skimage.morphology.disk(
            self.obs_dilation_selem_radius
        )

        # Goal setting
        self.set_goal_global = None
        self.curr_tracking_goal_global = None
        self.set_goal_pose_global = None

        # Recovery and tracking of state
        self.recovering = True
        self.recovery_heuristic = None
        self.blocked_steps_threshold = 2
        self.num_blocked_steps = 0
        self.prev_action = None
        self.prev_map_pose = None

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
            subgoal: tuple, bounding box of ROI specifying the subgoal in obs, i.e. (min_x, min_y, max_x, max_y)
            obs: observation from the simulator environment
        """
        min_x, min_y, max_x, max_y = subgoal
        depth = torch.tensor(
            np.transpose(obs[cam_uuid], axes=[2, 0, 1]), 
            device=self.device
        )
        sensor_config = sensor_config_dict[cam_uuid]
        _, sensor_yaw, _ = sensor_config.orientation

        # Get point cloud from depth map
        # TODO: Since the Habitat API provides different functions
        # to transform cloud based on pitch and yaw angles, we 
        # decompose the transformations into yaw (transform_pose_t)
        # --> pitch (transform_camera_view_t). This works because
        # currently pitch angle is always 0. This may not be correct
        # in other situations. To review and fix.
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
        points = self._rotate_yaw(points, sensor_yaw) # Account for camera rotation on base

        # Get bounding box
        bb_min_x, bb_max_x = torch.min(points[:, 0]).item(), torch.max(points[:, 0]).item()
        bb_min_y, bb_max_y = torch.min(points[:, 1]).item(), torch.max(points[:, 1]).item()
        bb_min_h, bb_max_h = torch.min(points[:, 2]).item(), torch.max(points[:, 2]).item()

        if self.visualise_subgoal_selection:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            scatter_points = point_cloud_base_coords[0].reshape(-1, 3)
            scatter_points = scatter_points[:, [1, 0, 2]] * torch.tensor([1., -1., 1.], device=self.device)
            scatter_points = self._rotate_yaw(scatter_points, sensor_yaw)
            scatter_points = scatter_points.cpu().numpy()
            scatter_points_crop = points.cpu().numpy()
            ax.scatter(
                scatter_points[:, 0], 
                scatter_points[:, 1], 
                scatter_points[:, 2],
                c=scatter_points[:, 2],
                cmap=plt.cm.viridis
            )
            ax.scatter(
                scatter_points_crop[:, 0], 
                scatter_points_crop[:, 1], 
                scatter_points_crop[:, 2],
                color='red',
                s=7**2,
            )
            ax.scatter(
                [(bb_max_x - bb_min_x) / 2],
                [(bb_max_y - bb_min_y) / 2],
                [(bb_max_h - bb_min_h) / 2],
                s=12**2,
                marker='x',
                color='cyan'
            )
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.savefig('logs/planning/pointcloud.png')
            plt.clf()

            plt.imshow(obs[cam_uuid].squeeze())
            rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.savefig("logs/planning/depth_crop_" + str(cam_uuid) + ".png")
            plt.clf()

        subgoal = [
            (bb_min_x + bb_max_x) / 2,
            (bb_min_y + bb_max_y) / 2,
        ]

        print("### Select image subgoal ###")
        print("Bounding box:", bb_min_x, bb_max_x, bb_min_y, bb_max_y, bb_min_h, bb_max_h)
        print("Image subgoal:", subgoal)

        self.set_subgoal_coord(subgoal, obs)

    def _rotate_yaw(self, XYZ, yaw_angle):
        R = ru.get_r_matrix([0.0, 0.0, 1.0], yaw_angle)
        XYZ = torch.matmul(
            XYZ.reshape(-1, 3), torch.from_numpy(R).float().transpose(1, 0).to(self.device)
        ).reshape(XYZ.shape)
        return XYZ

    def reset(self):
        # Reset subgoal
        self.reset_subgoal()

        # Reset debug stuff
        self.vis_dist_map = None
        self.goal_local_cell = None
        self.subgoal_local_cell = None
        self.set_goal_pose_local_cell = None

        # Reset state tracking and recovery stuff
        self.recovering = False
        self.num_blocked_steps = 0
        self.prev_action = None
        self.prev_map_pose = None

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
            # dist_map_vis[
            #     self.set_goal_pose_local_cell[0] - 1 : self.set_goal_pose_local_cell[0] + 2,
            #     self.set_goal_pose_local_cell[1] - 1 : self.set_goal_pose_local_cell[1] + 2,
            #     :2
            # ] = np.array([255, 255])

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

        # Check if we are stuck, and execute recovery if we are currently
        # still in a recovery mode
        if self.perform_recovery:
            if not self.recovering:
                self._check_if_stuck(px, py, po)

            if self.recovering:
                if self.recovery_heuristic is None:
                    print("Starting recovery!!")
                    self.num_blocked_steps = 0
                    self.recovery_heuristic = DiscreteRecovery()
                
                action, done, mode = self.recovery_heuristic.get_recovery_action([px, py, po])

                if done:
                    # If recovery has ended, reset the recovery heuristic, and
                    # pass through to normal planning again
                    self.recovering = False
                    self.recovery_heuristic = None
                    print("Recovery ended with:", mode)
                else:
                    # Recovery ongoing. Update state tracking, and output recovery action
                    self.prev_action = action
                    self.prev_map_pose = [px, py, po]
                    return action, False

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
                goal_tolerance=self.curr_goal_tolerance,
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
        # curr_goal_local_cell = self.mapper.global_pose_to_local_map_cell(
        #     self.curr_tracking_goal_global
        # )
        # init_goal_local_cell = self.mapper.global_pose_to_local_map_cell(
        #     self.set_goal_global
        # )
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
        stg_x, stg_y, replan, _ = self.planner.get_short_term_goal(
            curr_grid_pose,
            visualize=False
        )

        # TODO: Currently we compute our own stopping metric and ignore
        # the inaccurate stop from get_short_term_goal. To be fixed.
        tracking_goal_dist = torch.norm(
            self.curr_tracking_goal_global[:2] - torch.tensor([px, py], device=self.device)
        ).cpu().item()
        stop = tracking_goal_dist < self.curr_goal_tolerance

        if stop:
            print("Reached tracking goal!")

            init_goal_dist = torch.norm(
                self.set_goal_global[:2] - torch.tensor([px, py], device=self.device)
            ).cpu().item()
            if init_goal_dist > self.init_goal_tolerance:
                print("Failed to reach near init goal, with dist:", )
                # TODO: Should we trigger some kind of replanning?

            # Reset
            self.reset()

            return None, True
        
        if replan:
            # TODO: Unable to find a path to the goal. Need to execute
            # major recovery, probably including clearing the map.
            print("Replanning...")

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

        if self.set_goal_global is not None:
            init_goal_dist = torch.norm(
                self.set_goal_global[:2] - torch.tensor([px, py], device=self.device)
            ).cpu().item()
        init_goal_dist = None

        print(">>> Dist to init: ", init_goal_dist, "   Dist to tracking: ", tracking_goal_dist)

        # Update robot state tracking
        self.prev_action = action
        self.prev_map_pose = [px, py, po]

        return action, False
    
    def _check_if_stuck(self, px, py, ptheta):
        if self.prev_action is not None:
            prev_x, prev_y, prev_theta = self.prev_map_pose

            if type(self.prev_action) == str:
                # prev_action is Discrete

                if self.prev_action == 'move_forward':
                    # Check translational movement
                    self.num_blocked_steps = (
                        self.num_blocked_steps + 1 
                        if abs(prev_x - px) < 0.001 and abs(prev_y - py) < 0.001
                        else 0
                    )
                    if self.num_blocked_steps > 0:
                        print("### ", prev_x, px, prev_y, py)
                    if self.num_blocked_steps > self.blocked_steps_threshold:
                        self.recovering = True
                else:
                    # Check rotational movement (turning action executed)
                    self.num_blocked_steps = (
                        self.num_blocked_steps + 1
                        if abs(prev_theta - ptheta) < 0.001
                        else 0
                    )

                    if self.num_blocked_steps > 0:
                        print("@@@ ", prev_theta, ptheta, self.num_blocked_steps)
                    if self.num_blocked_steps > self.blocked_steps_threshold:
                        self.recovering = True

            else:
                # prev_action is a Continuous action
                raise NotImplementedError

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

    # config = setup_env_config()
    config = setup_env_config(default_config_path='configs/objectnav_hm3d_v2_with_semantic.yaml')
    env = ObjNavEnv(habitat.Env(config=config), config)
    obs = env.reset()

    # import habitat_sim
    # pos = [2.5770767, -0.34942314, 0.0]
    # ori = habitat_sim.utils.common.quat_from_angle_axis(np.pi, np.array([0, 0, 1]))
    # env.set_agent_position(pos, ori)
    # print("Set agent position")
    # print(env.env.sim.get_agent_state().position)

    import cv2
    import time
    cv2.namedWindow("Images")
    auto = False

    while True:
        # print(env.env.sim.get_agent_state().position)
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
                # [260, 380, 120, 360],
                [260, 120, 380, 360],
                "forward_depth",
                obs,
                get_camera_matrix(640, 480, 90)
            )

        if auto:
            action, stop = controller.step()
            if stop:
                auto = False
            print("(Auto) Action:", action)

        if action is not None:
            env.act(action)
