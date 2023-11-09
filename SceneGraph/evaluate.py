# from utils import VLM_BLIP, VLM_GDINO
# from PIL import Image
# '''
# TEST VLM Blip and GroundingDINO (GDINO):
# '''

# blip_model = VLM_BLIP()
# raw_image = Image.open("/home/zhanxin/Desktop/mount/EnvTest/test.png")
# output = blip_model.recoganize_obs(raw_image, "Where is the photo taken?")
# print(output)


# GDINO_model = VLM_GDINO()
# boxes, labels = GDINO_model.object_detect("/home/zhanxin/Desktop/mount/EnvTest/test.png")
# print('Object:', labels, boxes)

import pdb
from PIL import Image

import argparse
import os
# import dotenv
import json
import sys
from pathlib import Path
from collections import defaultdict

# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(0, '/home/zhanxin/Desktop/home-robot/src/home_robot')
sys.path.insert(0, '/home/zhanxin/Desktop/home-robot/src/home_robot_sim')
sys.path.insert(0, '/home/zhanxin/Desktop/home-robot/projects/habitat_objectnav/')

from habitat.config.default import get_config 
from habitat.core.env import Env
import habitat
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.agent.objectnav_agent.objectnav_agent_lang import ObjectNavAgent as ObjectNavAgentLang
from home_robot_sim.env.habitat_objectnav_env.habitat_objectnav_env import (
    HabitatObjectNavEnv,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an additional argument scene_ids that sets what scenes for this process to evaluate
    parser.add_argument(
        "--scene_ids",
        type=str,
        default="TEEsavR23oF,wcojb4TFT35",
        help="Comma separated list of scene ids to evaluate",
    )
    parser.add_argument(
        "--use_language",
        action="store_true",
        help="Use language",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="debug",
        help="Experiment name",
    )
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="/home/zhanxin/Desktop/SceneGraph/objectnav_hm3d.yaml",
        help="Path to config yaml",
    )
    # parser.add_argument(
    #     "--baseline_config_path",
    #     type=str,
    #     default="/home/zhanxin/Desktop/SceneGraph/objectnav_hm3d.yaml",
    #     help="Path to config yaml",
    # )
    parser.add_argument(
        "--max_episodes_per_scene",
        type=int,
        default=-1,
        help="Maximum number of episodes to evaluate per scene. Set to -1 for no limit.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        default="",
        help="openai api key",
    )
    parser.add_argument(
        "--openai_org",
        type=str,
        default="",
        help="openai api org",
    )
    parser.add_argument(
        "--print_images",
        action="store_true",
        help="Print images",
    )
    parser.add_argument(
        "--use_gt_semantics",
        action="store_true",
        help="Use ground truth semantics",
    )
    print("Arguments:")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

    config = get_config(args.habitat_config_path)
    config.defrost()
    print(config)

    config.NO_GPU = False
    config.VISUALIZE = False
    # config = get_config(args.habitat_config_path, args.baseline_config_path)
    # print(config[0].NUM_ENVIRONMENTS)
    # print(f"Using config: {args.baseline_config_path}")
    config.DUMP_LOCATION = '/home/zhanxin/Desktop/SceneGraph/trash'
    config.NUM_ENVIRONMENTS = 1

    print(f"Using {config.NUM_ENVIRONMENTS} environments")
    # config.GROUND_TRUTH_SEMANTICS = args.use_gt_semantics
    config.GROUND_TRUTH_SEMANTICS = 1
    print(f"Using ground truth semantics: {config.GROUND_TRUTH_SEMANTICS}")
    if config.GROUND_TRUTH_SEMANTICS == 0:
        if config.SEMANTIC_MODEL == "rednet":
            print("Using rednet")
        elif config.SEMANTIC_MODEL == "detic":
            print("Using detic")
    # config.PRINT_IMAGES = args.print_images
    config.PRINT_IMAGES = False

    print(f"Printing images: {config.PRINT_IMAGES}")

    config.EXP_NAME = args.experiment_name   # Set the experiment name from the argument
    print(f"Experiment name: {config.EXP_NAME}")
    
    args.use_language = True
    args.openai_key = 'test'
    args.openai_org = 'test'

    if not args.use_language:
        print('[Agent]: Normal ObjectNavAgent')
        agent = ObjectNavAgent(config=config)
    else:
        print('[Agent]: Language ObjectNavAgent')
        if args.openai_key == "":
            print("Loading openai key from .env")
            dotenv.load_dotenv(".env")
            config.OPENAI_KEY = os.getenv("OPENAI_API_KEY")
            config.OPENAI_ORG = os.getenv("OPENAI_ORG")
        else:
            print("Loading openai key from arguments")
            config.OPENAI_KEY = args.openai_key
            config.OPENAI_ORG = args.openai_org
        agent = ObjectNavAgentLang(config=config)
    config.freeze()

    # print(config)
    # pdb.set_trace()
    
    print('--- RUN THIS ---')
    # config2 = get_config('/home/zhanxin/Desktop/L3MVN/envs/habitat/configs/tasks/objectnav_hm3d.yaml')
    # test = habitat.RLEnv(config=habitat.get_config('/home/zhanxin/Desktop/L3MVN/envs/habitat/configs/tasks/objectnav_hm3d.yaml'))
    # print('--- RUN THIS2 ---')
    # testenv = Env(config=config2)
    # pdb.set_trace()
    # print('--- END THIS ---')
    env = HabitatObjectNavEnv(Env(config=config), config=config)

    # Create a file to dump the episode metrics
    episode_metrics_filename = f"datadump/{config.EXP_NAME}/episode_metrics.csv"
    # Make the directory if necessary
    Path(episode_metrics_filename).parent.mkdir(parents=True, exist_ok=True)
    # Write the header
    with open(episode_metrics_filename, "w") as f:
        f.write("id, episode_id, scene_id, goal_name, distance_to_goal, success, spl, soft_spl, distance_to_goal_reward\n")

    episodes_skiped = 0
    episode_counts = defaultdict(int)
    for i in range(len(env.habitat_env.episodes)):
        # try:
            agent.reset()
            env.reset()

            eval_scene = False
            for scene_id in args.scene_ids.split(","):
                if scene_id.strip() in env.habitat_env.current_episode.scene_id:
                    eval_scene = True
                    break

            if not eval_scene or (args.max_episodes_per_scene != -1 and episode_counts[env.habitat_env.current_episode.scene_id] >= args.max_episodes_per_scene):
                print(f"Skipping scene {env.habitat_env.current_episode.scene_id} episode {env.habitat_env.current_episode.episode_id}")
                continue

            t = 0
            print(f'[SCNEN ID]: {env.habitat_env.current_episode.scene_id}')
            while not env.episode_over:
                t += 1
                obs = env.get_observation()
                Image.fromarray(obs.rgb).save('./trash/'+ str(t)+'.png')

                action, info = agent.act(obs)

                print('[Execution]',action)
                if config.PRINT_IMAGES:
                    env.apply_action(action, info=info)
                else:
                    env.apply_action(action, info=None)

            episode_counts[env.habitat_env.current_episode.scene_id] += 1

            # Keys are "distance_to_goal", "success", "spl", "soft_spl", "distance_to_goal_reward"
            metrics = env.get_episode_metrics()
            # print(metrics)
            with open(episode_metrics_filename, "a") as f:
                f.write(f"{i}, {env.habitat_env.current_episode.episode_id}, {env.habitat_env.current_episode.scene_id}, {obs.task_observations['goal_name']}, {metrics['distance_to_goal']}, {metrics['success']}, {metrics['spl']}, {metrics['soft_spl']}, {metrics['distance_to_goal_reward']}\n")
        # except Exception as e:
        #     episodes_skiped += 1
        #     print(e)
        #     print("Skipping episode")

    print(f"Episodes skiped: {episodes_skiped}")
