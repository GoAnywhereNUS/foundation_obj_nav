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


class ObjNavEnv:
    def __init__(
        self, 
        habitat_env: habitat.core.env.Env, 
        config
    ):
        self.env = habitat_env
        self.config = config
        self._last_obs = None

    def reset(self):
        obs = self.env.reset()
        self._last_obs = obs
        return obs
  
    def set_agent_position(self, position, orientation):
        self.env.sim.set_agent_state(position, orientation)

    def episode_over(self) -> bool:
        return self.env.episode_over
  
    def get_episode_metrics(self) -> Dict:
        return self.env.get_metrics()
  
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
