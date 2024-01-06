import argparse
import os
# import dotenv
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

import yaml
from habitat.core.env import Env
import habitat
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from homerobot_utils.objectnav_agent_lang import ObjectNavAgent as ObjectNavAgentLang
from home_robot_sim.env.habitat_objectnav_env.habitat_objectnav_env import (
    HabitatObjectNavEnv,
)
from habitat.config.default import get_config, get_agent_config
from habitat.config import read_write

#######################################
########## Used for GUI issue #########
#######################################
#######################################
# import cv2
# for k, v in os.environ.items():
#     if k.startswith("QT_") and "cv2" in v:
#         del os.environ[k]
        

from habitat.config.default_structured_configs import (
    HabitatSimRGBSensorConfig,
    HabitatSimDepthSensorConfig,
    HabitatSimSemanticSensorConfig,
    ThirdRGBSensorConfig,
    ThirdDepthSensorConfig,
    # HeadPanopticSensorConfig,    
)

from dataclasses import dataclass
@dataclass
class ThirdSemanticSensorConfig(HabitatSimSemanticSensorConfig):
    uuid: str = "third_semantic"
    width: int = 256
    height: int = 256

def get_sensor_config(config, sensor):
    assert ('rgb' in sensor
            or 'depth' in sensor
            or 'semantic' in sensor), \
         "Invalid sensor type: " + sensor

    if 'rgb' in sensor:
        return ThirdRGBSensorConfig(
                height=config.ENVIRONMENT.frame_height,
                width=config.ENVIRONMENT.frame_width,
                position=[0, config.ENVIRONMENT.camera_height, 0],
                orientation=[0, 0, 0],
                sensor_subtype="PINHOLE",
                hfov=config.ENVIRONMENT.hfov,
                uuid=sensor,
        )
    elif 'depth' in sensor:
        return ThirdDepthSensorConfig(
                height=config.ENVIRONMENT.frame_height,
                width=config.ENVIRONMENT.frame_width,
                position=[0, config.ENVIRONMENT.camera_height, 0],
                orientation=[0, 0, 0],
                sensor_subtype="PINHOLE",
                hfov=config.ENVIRONMENT.hfov,
                min_depth=config.ENVIRONMENT.min_depth,
                max_depth=config.ENVIRONMENT.max_depth,
                normalize_depth=True,
                uuid=sensor,
        )
    elif 'semantic' in sensor:
        return ThirdSemanticSensorConfig(
                height=config.ENVIRONMENT.frame_height,
                width=config.ENVIRONMENT.frame_width,
                position=[0, config.ENVIRONMENT.camera_height, 0],
                orientation=[0, 0, 0],
                sensor_subtype="PINHOLE",
                hfov=config.ENVIRONMENT.hfov,
                uuid=sensor,
        )
    else:
        # Should not reach here
        raise Exception

if __name__ == "__main__":
    print("-" * 100)
    test_scene_ids = '' #   use common to seperate,  default="TEEsavR23oF,wcojb4TFT35"
    max_episodes_per_scene = 10  # Maximum number of episodes to evaluate per scene. Set to -1 for no limit
    openai_key_path = '../../configs/openai_api_key.yaml'
    env_config_path = "configs/LFG_hm3d_eval.yaml"
    use_semantics = True

    config = get_config(env_config_path)

    with read_write(config):
        # Update dataset parameters
        config.EXP_NAME = 'debug'   # Set the experiment name from the argument
        config.NUM_ENVIRONMENTS = 1
        config.VISUALIZE = False
        config.PRINT_IMAGES = True
        config.NUM_ENVIRONMENTS = 1
        config.GROUND_TRUTH_SEMANTICS = 1 # 1 for use ground truth, 0 for detection model


        print(f"Experiment name: {config.EXP_NAME}")
        print(f"Printing images: {config.PRINT_IMAGES}")
        print(f"Using config: {env_config_path}")
        print(f"Using {config.NUM_ENVIRONMENTS} environments")
        print(f"Using ground truth semantics: {config.GROUND_TRUTH_SEMANTICS}")
        
        if config.GROUND_TRUTH_SEMANTICS == 0:
            if config.SEMANTIC_MODEL == "rednet":
                print("Using rednet")
            elif config.SEMANTIC_MODEL == "detic":
                print("Using detic")

        # Set openai key
        with open(openai_key_path, 'r') as f:
            key_dict = yaml.safe_load(f)
            config.OPENAI_KEY = key_dict['api_key']

        # Update Sensor
        config.habitat.simulator.agents.main_agent.sim_sensors = {
            sensor : get_sensor_config(config, sensor)
            for sensor in config.sensor_setup
        }

    if not use_semantics:
        agent = ObjectNavAgent(config=config)
    else:
        agent = ObjectNavAgentLang(config=config)
    
    env = HabitatObjectNavEnv(Env(config=config), config=config)


    # Create a file to dump the episode metrics
    episode_metrics_filename = f"datadump/{config.EXP_NAME}/episode_metrics.csv"

    Path(episode_metrics_filename).parent.mkdir(parents=True, exist_ok=True)  # Make the directory if necessary

    with open(episode_metrics_filename, "w") as f:
        f.write("id, episode_id, scene_id, goal_name, distance_to_goal, success, spl, soft_spl, distance_to_goal_reward\n") # Write the header

    episodes_skiped = 0
    episode_counts = defaultdict(int)
    
    for i in range(len(env.habitat_env.episodes)):
        # try:
        agent.reset()
        env.reset()

        eval_scene = False
        for scene_id in test_scene_ids.split(","):
            if scene_id.strip() in env.habitat_env.current_episode.scene_id:
                eval_scene = True
                break

        if not eval_scene or (max_episodes_per_scene != -1 and episode_counts[env.habitat_env.current_episode.scene_id] >= max_episodes_per_scene):
            print(f"Skipping scene {env.habitat_env.current_episode.scene_id} episode {env.habitat_env.current_episode.episode_id}")
            continue

        t = 0
        while not env.episode_over:
            t += 1
            obs = env.get_observation()
            action, info = agent.act(obs)
            info['semantic_map_config'] = config.AGENT.SEMANTIC_MAP
            if config.PRINT_IMAGES:
                env.apply_action(action, info=info)
            else:
                env.apply_action(action, info=None)

        episode_counts[env.habitat_env.current_episode.scene_id] += 1

        # Keys are "distance_to_goal", "success", "spl", "soft_spl", "distance_to_goal_reward"
        metrics = env.get_episode_metrics()
        print(metrics)
        with open(episode_metrics_filename, "a") as f:
            f.write(f"{i}, {env.habitat_env.current_episode.episode_id}, {env.habitat_env.current_episode.scene_id}, {obs.task_observations['goal_name']}, {metrics['distance_to_goal']}, {metrics['success']}, {metrics['spl']}, {metrics['soft_spl']}, {metrics['distance_to_goal_reward']}\n")
        # except Exception as e:
        #     episodes_skiped += 1
        #     print(e)
        #     print("Skipping episode")

    print(f"Episodes skiped: {episodes_skiped}")