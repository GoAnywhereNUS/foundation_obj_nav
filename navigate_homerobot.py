import os
import sys
import numpy as np
import re
from PIL import Image
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from navigator import Navigator
from scene_graph import SceneGraph, default_scene_graph_specs
import cv2
import pdb

import math
from utils.mapper import Mapper
from utils.habitat_utils import ObjNavEnv, setup_env_config

import habitat
from fmm_controller import *
import json
import random
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NavigatorHomeRobot(Navigator):
    def __init__(
        self,
        scene_graph_specs=default_scene_graph_specs,
        llm_config_path="configs/gpt_config.yaml",
        task_config_path = 'configs/objectnav_hm3d_v2_with_semantic.yaml',
        data_path = "configs/homerobot_hm3d_objectnav.yaml"
    ):
        
        super().__init__(
            scene_graph_specs=scene_graph_specs, 
            llm_config_path=llm_config_path,
            visualise=True
        )

        self.GT = True

        # Setup home robot sim environment
        config = setup_env_config(params_path = data_path, default_config_path= task_config_path)
        self.config = config
        self.env = ObjNavEnv(habitat.Env(config=config), config)

        self.turn_left_amount = 30
        for i in self.env.env.sim.config.agents[0].action_space.keys():
            if self.env.env.sim.config.agents[0].action_space[i].name =='turn_left':
                self.turn_left_amount = self.env.env.sim.config.agents[0].action_space[i].actuation.amount

        if 'hm3d' in task_config_path:
            self.dataset = 'hm3d'
            self.goal = self.env.env.current_episode.object_category
        elif 'gibson' in task_config_path:
            self.dataset = 'gibson'
            self.GT = False
            data_path = self.config.habitat.dataset.data_path[:self.config.habitat.dataset.data_path.find('val.json.gz')]
            scnen_path = self.env.env.current_episode.scene_id
            cur_episode_id = self.env.env.current_episode.episode_id
            scene_name = scnen_path[scnen_path.rfind('/')+1:scnen_path.rfind('.glb')]
            episode_path = f'{data_path}content/{scene_name}_episodes.json.gz'
            import gzip 
            f = gzip.open(episode_path, "rt") 

            dataset_info_path = f'{data_path}val_info.pbz2'
            deserialized = json.loads(f.read())
            episode_info = deserialized['episodes'][cur_episode_id]
            self.goal = episode_info["object_category"]

            self.env.metric_info = {'scene_name': scene_name, 'dataset_info_path': dataset_info_path, 'episode_info': episode_info}
            

        env_semantic_names = [s.category.name().lower() if s != None else 'Unknown' for s in self.env.env.sim.semantic_annotations().objects]
        env_semantic_names = ['sofa' if x == 'couch' else x for x in env_semantic_names]
        env_semantic_names = ['toilet' if x == 'toilet seat' else x for x in env_semantic_names]
        self.semantic_annotations = env_semantic_names
    
        if self.goal == "tv_monitor":
            self.goal = "tv"
        elif self.goal == "potted plant":
            self.goal = "plant"
        # Set up controller
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.controller = FMMController(self.device, env_config=self.config)

        selected_traversable_categories = []
        # for i in ["stair", "stairs"]:
        #     if i in env_semantic_names:
        #         selected_traversable_categories.append(i)

        self.controller = FMMController(
            self.device, 
            env_config=self.config,
            # semantic_categories=["others"] + selected_traversable_categories,
            # semantic_annotations=env_semantic_names,
            # traversable_categories=selected_traversable_categories,
        )

            # Set up flags
        self.controller_active = False

        self.env.reset()

        # TODO: Review these variables
        # self.defined_entrance = ['doorway', 'entrance', 'door frame']


    def reset(self):
        super().reset()

        # Reset environment
        self.env.reset() # TODO: Is this needed?
        env_semantic_names = [s.category.name().lower() if s != None else 'Unknown' for s in self.env.env.sim.semantic_annotations().objects]
        env_semantic_names = ['sofa' if x == 'couch' else x for x in env_semantic_names]
        env_semantic_names = ['toilet' if x == 'toilet seat' else x for x in env_semantic_names]
        self.semantic_annotations = env_semantic_names

        self.turn_left_amount = 30
        for i in self.env.env.sim.config.agents[0].action_space.keys():
            if self.env.env.sim.config.agents[0].action_space[i].name =='turn_left':
                self.turn_left_amount = self.env.env.sim.config.agents[0].action_space[i].actuation.amount

        if 'hm3d' in self.dataset:
            self.goal = self.env.env.current_episode.object_category
        elif 'gibson' in self.dataset:
            self.GT = False
            data_path = self.config.habitat.dataset.data_path[:self.config.habitat.dataset.data_path.find('val.json.gz')]
            scnen_path = self.env.env.current_episode.scene_id
            cur_episode_id = self.env.env.current_episode.episode_id
            scene_name = scnen_path[scnen_path.rfind('/')+1:scnen_path.rfind('.glb')]
            episode_path = f'{data_path}content/{scene_name}_episodes.json.gz'
            import gzip 
            f = gzip.open(episode_path, "rt") 

            dataset_info_path = f'{data_path}val_info.pbz2'
            deserialized = json.loads(f.read())
            episode_info = deserialized['episodes'][cur_episode_id]
            self.goal = episode_info["object_category"]

            self.env.metric_info = {'scene_name': scene_name, 'dataset_info_path': dataset_info_path, 'episode_info': episode_info}
          
        if self.goal == "tv_monitor":
            self.goal = "tv"
        elif self.goal == "potted plant":
            self.goal = "plant"
        # Reset controller
       
        # self.controller = FMMController(self.device, env_config=self.config)
        selected_traversable_categories = []
        for i in ["stair", "stairs"]:
            if i in env_semantic_names:
                selected_traversable_categories.append(i)

        self.controller = FMMController(
            self.device, 
            env_config=self.config,
            semantic_categories=["others"] + selected_traversable_categories,
            semantic_annotations=env_semantic_names,
            traversable_categories=selected_traversable_categories,
        )

        # Reset flags
        self.controller_active = False


    def _observe(self):
        """
        Get observations from the environment (e.g. render images for
        an agent in sim, or take images at current location on real robot).
        To be overridden in subclass.

        Return:
            images: dict of images taken at current pose
        """
        obs = self.env.get_observation()
        # obs['sensor_label'] = ['left', 'forward', 'right', 'rear']
        if not self.controller_active:
            new_obs = {}
            direction = ['left', 'rear', 'right','forward']
            for dir in direction:
                turn_round  = int(360/len(direction)/self.turn_left_amount)
                for i in range(turn_round):
                    self.env.act('turn_left')
                    self.action_step += 1
                tmp_obs = self.env.get_observation()
                new_obs[dir+'_rgb'] = tmp_obs['forward_rgb']
                new_obs[dir+'_depth'] = tmp_obs['forward_depth']
                new_obs[dir+'_semantic'] = tmp_obs['forward_semantic']
            new_obs['gps'] = obs['gps']
            new_obs['compass'] = obs['compass']
            return new_obs
        else:
            return obs
    
    def run(self):
        """
        Executes the navigation loop. To be implemented in each
        Navigator subclass.

        For NavigatorHomeRobot, run() executes the loop for a single
        episode in a specific scene.

        Returns:
            None
        """
        self.action_logging.write(f'[EPISODE ID]: {self.env.env.current_episode.episode_id}\n')
        self.action_logging.write(f'[SCENE ID]: {self.env.env.current_episode.scene_id}\n')
        self.action_logging.write(f'[GOAL]: {self.goal}\n')
        self.action_logging.write(f'[Ground Truth Sem]: {self.GT}\n')
        self.action_logging.close()

        cv2.namedWindow("View")

        while (self.action_step < self.max_episode_step) and (self.llm_loop_iter <= 40):
            self.action_logging = open(self.action_log_path, 'a')
            obs = self._observe()
            self.controller.update(obs)
            images = self.controller.visualise(obs)
            self.env.update_path_distance()
            cv2.imshow("View", images)
            cv2.waitKey(10)

            # High-level perception-reasoning. Run when we have 
            # paused and are awaiting next instructions.
            subgoal_position = None
            if not self.controller_active:
                preprocessed_obs = {'forward': obs['forward_rgb'], 'right': obs['right_rgb'], 'left': obs['left_rgb'], 'rear': obs['rear_rgb'], 'info':{'forward_depth':obs['forward_depth'], 'left_depth':obs['left_depth'], 'right_depth':obs['right_depth'], 'rear_depth':obs['rear_depth'],'forward_semantic':obs['forward_semantic'], 'left_semantic':obs['left_semantic'], 'right_semantic':obs['right_semantic'], 'rear_semantic':obs['rear_semantic'] } } 
                subgoal_position, cam_uuid = self.loop(preprocessed_obs)
                if self.check_goal(self.last_subgoal) and self.success_flag:
                    self.action_logging.write(f"[END]: SUCCESS Checked! \n")
                    break
                elif (not self.check_goal(self.last_subgoal)) and self.success_flag:
                    self.success_flag = False
                print('Set Subgoals:',subgoal_position)
                if subgoal_position is not None:
                    if self.dataset == 'gibson' and  ('television' in self.last_subgoal or 'tv' in self.last_subgoal):
                        subgoal_position[1] += 150
                        subgoal_position[3] += 150
                    try:
                        self.controller.set_subgoal_image(
                            subgoal_position,
                            cam_uuid + '_depth',
                            obs,
                            get_camera_matrix(640, 480, 90)
                        )
                        self.controller_active = True
                    except:
                        self.action_logging.write(f'ERROR: Cannot set subgoal to controller {subgoal_position}\n')
                        self.llm_loop_iter += 1
            # Low-level perception-reasoning. Run all the time.
            # TODO:

            # Update and handle controller. Set subgoal if
            # issued by perception-reasoning, otherwise continue
            # to navigate with the controller

            if self.controller_active:
                action, success = self.controller.step()
                print('Control', action, success)
                if action is None:
                    self.controller_active = False
                    if self.check_goal(self.last_subgoal) and success and 'floor' not in self.last_subgoal:
                        self.success_flag = True
                        self.action_logging.write(f"[END]: SUCCESS\n")
                else:
                    print("(Auto) Action:", action)
                    self.env.act(action)
                    self.action_step += 1

                    self.action_logging.write(f"[Action]: {action}\n")
                    self.action_logging.write(f'[Pos]: {self.env.env.sim.agents[0].get_state().position} [Rotation]: {self.env.env.sim.agents[0].get_state().rotation} \n')

                    if self.action_step >= self.max_episode_step:
                        self.action_logging.write(f"[END]: FAIL\n")

                # TODO: Does any feedback need to be given to perception-reasoning?
            self.action_logging.close()
            print('Pos:', self.env.env.sim.agents[0].get_state().position)
            
        # Record ending position
        self.action_logging = open(self.action_log_path, 'a')
        preprocessed_obs = {'forward': obs['forward_rgb'], 'right': obs['right_rgb'], 'left': obs['left_rgb'], 'rear': obs['rear_rgb'], 'info':{'forward_depth':obs['forward_depth'], 'left_depth':obs['left_depth'], 'right_depth':obs['right_depth'], 'rear_depth':obs['rear_depth'],'forward_semantic':obs['forward_semantic'], 'left_semantic':obs['left_semantic'], 'right_semantic':obs['right_semantic'], 'rear_semantic':obs['rear_semantic'] } } 
        img_lang_obs = self.perceive(preprocessed_obs)
        if self.visualisation:
            self.visualise_objects(preprocessed_obs, img_lang_obs)            
    
        while not self.env.env.episode_over:
            self.env.act('stop')

        for i, obj in enumerate(self.semantic_annotations):
            if self.check_goal(obj):
                self.action_logging.write(f"[Goal Check]: id :{i}; obj: {obj}, pos:{self.env.env.sim.semantic_annotations().objects[i].aabb.center} \n")

        print(self.env.get_episode_metrics())
        self.action_logging.write(f"[Goal Synonyms]: {self.goal_synonyms}\n")
        self.action_logging.write(f"[Metrics]: {self.env.get_episode_metrics()}\n")
        self.action_logging.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    nav = NavigatorHomeRobot(task_config_path = 'configs/objectnav_hm3d_v2_with_semantic.yaml', data_path = "configs/homerobot_hm3d_objectnav.yaml")
    test_episode = 10
    test_history = []
    # str(i) for i in range(test_episode)
    while True:
        scnen_path = nav.env.env.current_episode.scene_id
        scene_name = scnen_path[scnen_path.rfind('/')+1:scnen_path.rfind('.basis')]
        # episode = rerun_case[scene_name]
        while str(nav.env.env.current_episode.episode_id) not in ['1'] or ((nav.env.env.current_episode.scene_id + str(nav.env.env.current_episode.episode_id) ) in test_history):
            try:
                print('RESET', nav.env.env.current_episode.episode_id,nav.env.env.current_episode.scene_id )
                nav.reset()
            except:
                sys.exit(0)
        test_history.append(nav.env.env.current_episode.scene_id + str(nav.env.env.current_episode.episode_id) )
        # try:
        nav.run()
        # except:
        #     print('error')