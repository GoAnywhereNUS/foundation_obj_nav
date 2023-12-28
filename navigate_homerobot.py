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
from utils.fmm_planner import FMMPlanner
from utils.habitat_utils import ObjNavEnv, setup_env_config

import habitat
from fmm_controller import *

import random
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NavigatorHomeRobot(Navigator):
    def __init__(
        self,
        scene_graph_specs=default_scene_graph_specs,
        llm_config_path="configs/gpt_config.yaml"
    ):
        
        super().__init__(
            scene_graph_specs=scene_graph_specs, 
            llm_config_path=llm_config_path
        )

        # Setup home robot sim environment
        config = setup_env_config(default_config_path='configs/objectnav_hm3d_v2_with_semantic.yaml')
        self.config = config
        self.env = ObjNavEnv(habitat.Env(config=config), config)

        env_semantic_names = [s.category.name().lower() for s in self.env.env.sim.semantic_annotations().objects]
        env_semantic_names = ['sofa' if x == 'couch' else x for x in env_semantic_names]
        env_semantic_names = ['toilet' if x == 'toilet seat' else x for x in env_semantic_names]
        self.semantic_annotations = env_semantic_names

        # Set up controller
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.controller = FMMController(self.device, env_config=self.config)

        # Set up flags
        self.controller_active = False

        # TODO: Review these variables
        # self.defined_entrance = ['doorway', 'entrance', 'door frame']


    def reset(self):
        super().reset()

        # Reset environment
        self.env.reset() # TODO: Is this needed?
        env_semantic_names = [s.category.name().lower() for s in self.env.env.sim.semantic_annotations().objects]
        env_semantic_names = ['sofa' if x == 'couch' else x for x in env_semantic_names]
        env_semantic_names = ['toilet' if x == 'toilet seat' else x for x in env_semantic_names]
        self.semantic_annotations = env_semantic_names
        self.goal = self.env.env.current_episode.object_category

        # Reset controller
        self.controller = FMMController(self.device, env_config=self.config)

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

        cv2.namedWindow("View")

        while (self.action_step < self.max_episode_step) and (not self.success_flag):
            obs = self._observe()
            self.controller.update(obs)
            images = self.controller.visualise(obs)
            cv2.imshow("View", images)
            cv2.waitKey(10)

            # High-level perception-reasoning. Run when we have 
            # paused and are awaiting next instructions.
            subgoal_position = None
            if not self.controller_active:
                subgoal_position, cam_uuid = self.loop(obs)

                if subgoal_position is not None:
                    self.controller.set_subgoal_image(
                        subgoal_position,
                        cam_uuid,
                        obs,
                        get_camera_matrix(640, 480, 90)
                    )
                    self.controller_active = True

            # Low-level perception-reasoning. Run all the time.
            # TODO:

            # Update and handle controller. Set subgoal if
            # issued by perception-reasoning, otherwise continue
            # to navigate with the controller

            if self.controller_active:
                action, success = self.controller.step()
                if action is None:
                    self.controller_active = False
                    if self.last_subgoal == self.goal:
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
            
            print('Pos:', self.env.env.sim.agents[0].get_state().position)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    nav = NavigatorHomeRobot()
    nav.reset()
    test_episode = 2
    test_history = []
    while True:
        while nav.env.env.current_episode.episode_id not in [str(i) for i in range(test_episode)] or ((nav.env.env.current_episode.scene_id + nav.env.env.current_episode.episode_id ) in test_history):
            try:
                print('RESET', nav.env.env.current_episode.episode_id,nav.env.env.current_episode.scene_id )
                nav.reset()
            except:
                sys.exit(0)
        test_history.append(nav.env.env.current_episode.scene_id + nav.env.env.current_episode.episode_id )
        nav.run()