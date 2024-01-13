import yaml
import numpy as np
from typing import Any, Dict, Optional, TypeVar

import habitat
from habitat.config import read_write
from habitat.config.default import get_config, get_agent_config
from habitat.config.default_structured_configs import (
    HabitatSimRGBSensorConfig,
    HabitatSimDepthSensorConfig,
    HabitatSimSemanticSensorConfig,
    ThirdRGBSensorConfig,
    ThirdDepthSensorConfig,
    # HeadPanopticSensorConfig,    
)

from gym import spaces
from .fmm_planner import *
import skimage
import quaternion

ActType = TypeVar("ActType")

from dataclasses import dataclass
@dataclass
class ThirdSemanticSensorConfig(HabitatSimSemanticSensorConfig):
    uuid: str = "third_semantic"
    width: int = 256
    height: int = 256

sensor_config_dict = {
            # 'forward_rgb': HabitatSimRGBSensorConfig(
            #     height=480,
            #     width=640,
            #     position=[0, 0.88, 0],
            #     orientation=[0, 0, 0],
            #     sensor_subtype="PINHOLE",
            #     hfov=90
            # ),
            # Smemantic
            'left_semantic': ThirdSemanticSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0, np.pi/2, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                uuid="left_semantic",
            ),
            'forward_semantic': ThirdSemanticSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0, 0, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                uuid="forward_semantic",
            ),
            'right_semantic': ThirdSemanticSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0, -np.pi/2, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                uuid="right_semantic",
            ),
            'rear_semantic': ThirdSemanticSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0,  -np.pi, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                uuid="rear_semantic",
            ),
            # RGB
            'left_rgb': ThirdRGBSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0, np.pi/2, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                uuid="left_rgb",
            ),
            'forward_rgb': ThirdRGBSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0, 0, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                uuid="forward_rgb",
            ),
            'right_rgb': ThirdRGBSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0, -np.pi/2, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                uuid="right_rgb",
            ),
            'rear_rgb': ThirdRGBSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0,  -np.pi, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                uuid="rear_rgb",
            ),
            # 'forward_depth': HabitatSimDepthSensorConfig(
            #     height=480,
            #     width=640,
            #     position=[0, 0.88, 0],
            #     orientation=[0, 0, 0],
            #     sensor_subtype="PINHOLE",
            #     hfov=90,
            #     min_depth=0.0,
            #     max_depth=10.0,
            #     normalize_depth=False,
            # ),
            'left_depth': ThirdDepthSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0, np.pi/2, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                min_depth=0.0,
                max_depth=15.0,
                normalize_depth=False,
                uuid="left_depth",
            ),
            'forward_depth': ThirdDepthSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0, 0, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                min_depth=0.0,
                max_depth=10.0,
                normalize_depth=False,
                uuid="forward_depth",
            ),
            'right_depth': ThirdDepthSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0, -np.pi/2, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                min_depth=0.0,
                max_depth=15.0,
                normalize_depth=False,
                uuid="right_depth",
            ),
            'rear_depth': ThirdDepthSensorConfig(
                height=480,
                width=640,
                position=[0, 0.88, 0],
                orientation=[0, -np.pi, 0],
                sensor_subtype="PINHOLE",
                hfov=90,
                min_depth=0.0,
                max_depth=15.0,
                normalize_depth=False,
                uuid="rear_depth",
            )
        }


def setup_env_config(
        params_path="configs/homerobot_objectnav.yaml",
        default_config_path="home-robot/src/third_party/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d_v2_with_semantic.yaml"
    ):

    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    config = habitat.get_config(default_config_path)

    with read_write(config):
        # Update dataset parameters
        config.habitat.dataset.data_path = params['habitat_data']['metadata_path']
        config.habitat.dataset.content_scenes = params['habitat_data']['content_scenes']
        config.habitat.dataset.scenes_dir = params['habitat_data']['scenes_dir']

        # Update the env parameters
        config.habitat.environment.max_episode_steps = 999999999999

        # Update agent parameters
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.update({
            id : sensor_config_dict[id]
            for id in params['sensor_setup']
        })

    return config


def get_sim_location(agent_state):
    """Returns x, y, o pose of the agent in the Habitat simulator."""
    x = -agent_state.position[2]
    y = -agent_state.position[0]
    axis = quaternion.as_euler_angles(agent_state.rotation)[0]
    if (axis % (2 * np.pi)) < 0.1 or (axis %
                                    (2 * np.pi)) > 2 * np.pi - 0.1:
        o = quaternion.as_euler_angles(agent_state.rotation)[1]
    else:
        o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o

def sim_continuous_to_sim_map(sim_loc, map_obj_origin):
    """Converts absolute Habitat simulator pose to ground-truth 2D Map
    coordinates.
    """
    x, y, o = sim_loc
    min_x, min_y = map_obj_origin / 100.0
    x, y = int((-x - min_x) * 20.), int((-y - min_y) * 20.)

    o = np.rad2deg(o) + 180.0
    return y, x, o


def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1

    return dx, dy, do


def get_new_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change

    global_dx = dx * np.sin(np.deg2rad(o)) + dy * np.cos(np.deg2rad(o))
    global_dy = dx * np.cos(np.deg2rad(o)) - dy * np.sin(np.deg2rad(o))
    x += global_dy
    y += global_dx
    o += np.rad2deg(do)
    if o > 180.:
        o -= 360.
    return x, y, o

def get_pose_change(curr_sim_pose, last_sim_location):
    """Returns dx, dy, do pose change of the agent relative to the last
    timestep."""
    dx, dy, do = get_rel_pose_change(
        curr_sim_pose, last_sim_location)
    return dx, dy, do


class ObjNavEnv:
    def __init__(
        self, 
        habitat_env: habitat.core.env.Env, 
        config
    ):
        self.env = habitat_env
        self.config = config
        self._last_obs = None
        self.path_length = 1e-5
        self.metric_info = None
        self.last_pos = None

    def reset(self):
        obs = self.env.reset()
        self._last_obs = obs
        self.path_length = 1e-5
        self.last_pos = None      
        return obs
  
    def set_agent_position(self, position, orientation):
        self.env.sim.set_agent_state(position, orientation)

    def episode_over(self) -> bool:
        return self.env.episode_over
  
    def get_episode_metrics(self) -> Dict:
        if self.metric_info != None:
            scnen_name = self.metric_info['scene_name']
            dataset_info_path = self.metric_info['dataset_info_path']
            
            import bz2
            import _pickle as cPickle
            f = bz2.BZ2File(dataset_info_path, 'rb')
            dataset_info = cPickle.load(f)
            scene_info = dataset_info[scnen_name]

            episode =  self.metric_info['episode_info']
            goal_idx = episode["object_id"]
            floor_idx = episode["floor_id"]
            sem_map = scene_info[floor_idx]['sem_map']
            map_obj_origin = scene_info[floor_idx]['origin']
            object_boundary = 1.0
            map_resolution = 5

            selem = skimage.morphology.disk(2)
            traversible = skimage.morphology.binary_dilation(sem_map[0], selem) != True
            traversible = 1 - traversible
            planner = FMMPlanner(traversible)
            selem = skimage.morphology.disk(int(object_boundary * 100. / map_resolution))
            goal_map = skimage.morphology.binary_dilation(sem_map[goal_idx + 1], selem) != True
            goal_map = 1 - goal_map
            planner.set_multi_goal(goal_map)
            curr_loc = sim_continuous_to_sim_map(get_sim_location(self.env.sim.agents[0].get_state()), map_obj_origin)
            dist = planner.fmm_dist[curr_loc[0], curr_loc[1]] / 20.0
            if dist == 0.0:
                success = 1
            else:
                success = 0
            pos = episode["start_position"]
            x = -pos[2]
            y = -pos[0]
            min_x, min_y = map_obj_origin / 100.0
            map_loc = int((-y - min_y) * 20.), int((-x - min_x) * 20.)
            starting_loc = map_loc
            starting_distance = planner.fmm_dist[starting_loc] / 20.0 + object_boundary
            spl = min(success * starting_distance / self.path_length, 1)
            print(starting_distance, self.path_length)
            return {'success':success, 'spl':spl, 'distance_to_goal': dist}
        else:
            return self.env.get_metrics()
    
    def update_path_distance(self):
        if self.metric_info == None:
            return
        else:
            cur_pos = get_sim_location(self.env.sim.agents[0].get_state())
            if self.last_pos != None:
                dx, dy, do = get_pose_change(cur_pos, self.last_pos)
                self.path_length +=  get_l2_distance(0, dx, 0, dy)
                self.last_pos = cur_pos
                print('path length', self.path_length)
            else:
                self.last_pos = cur_pos

    def act(self, action):
        obs = self.env.step(action)
        self._last_obs = obs

    def get_observation(self):
        return self._last_obs

    @property
    def observation_space(self):
        return self.env.observation_space

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    @property
    def action_space(self) -> spaces.Space[ActType]:
        return self.env.action_space

    def current_episode(self, all_info: bool = True) -> int:
        return self.env.current_episode(all_info)

    @property
    def number_of_episodes(self) -> int:
        return self.env.number_of_episodes

    @property
    def original_action_space(self) -> spaces.space:
        return self.env.original_action_space
