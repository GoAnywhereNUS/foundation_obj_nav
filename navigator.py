import os
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mapper import OSGMapper
from reasoner import Reasoner
from utils.vis_utils import Visualiser
import json

class Navigator:
    def __init__(
        self,
        scene_graph_specs=None,
        llm_config_path="configs/gpt_config.yaml",
        visualise=False,
    ):
        # Set up parameters to be used in reasoning
        self.llm_max_query = 10
        self.llm_sampling_query = 5
        self.explored_node = []
        self.path = []
        self.last_subgoal = None
        self.goal = None

        self.action_step = 0
        self.max_episode_step = 500
        self.llm_loop_iter = 0
        self.is_navigating = False
        self.success_flag = False
        self.GT = False
        self.semantic_annotations = None
        self.goal_synonyms = None
        self.perceive_filter = False

        self.trial_folder = self.create_log_folder()
        self.action_log_path = os.path.join(self.trial_folder, 'action.txt')
        self.action_logging = open(self.action_log_path, 'a')

        # Note: Env and controller to be implemented in subclass

        # TODO: Review these variables and parameters.
        self.defined_entrance = ['door', 'doorway', 'doorframe', 'window']

        # Visualisation
        self.visualisation = visualise 
        self.visualiser = Visualiser() if self.visualisation else None
        self.vis_image = np.ones((100, 100, 3))

        self.mapper = OSGMapper()
        self.reasoner = Reasoner(spec=self.mapper.spec, action_logger=self.action_logging)

        self.tmp_path = []

    def reset(self):
        # Reset data structures
        self.explored_node = []
        self.path = []
        self.last_subgoal = None
        self.goal = None

        # Reset variables
        self.action_step = 0
        self.llm_loop_iter = 0
        self.is_navigating = False
        self.success_flag = False
        self.goal_synonyms = None
        self.perceive_filter = False

        # Reset logging
        self.trial_folder = self.create_log_folder()
        self.action_log_path = os.path.join(self.trial_folder, 'action.txt')
        self.action_logging = open(self.action_log_path, 'a')

    def create_log_folder(log_folder = 'logs'):
        logs_folder = 'logs'

        # Filter folders that start with "trial_"
        all_folders = [folder for folder in os.listdir(logs_folder) if os.path.isdir(os.path.join(logs_folder, folder))]
        trial_folders = [folder for folder in all_folders if folder.startswith("trial_")]

        # Extract the numbers and find the maximum
        numbers = [int(folder.split("_")[1]) for folder in trial_folders]
        max_number = max(numbers, default=0)
        trial_folder = os.path.join(logs_folder, 'trial_' + str(max_number))

        trial = max_number
        if os.path.exists(trial_folder):
            if len(os.listdir(trial_folder)) == 0:
                # Folder has no data, so we can go ahead and use it
                return trial_folder
            elif len(os.listdir(trial_folder)) == 1:
                action_file_path = os.path.join(trial_folder, os.listdir(trial_folder)[0])
                if os.path.getsize(action_file_path) == 0:
                # Folder has an empty file (action.txt), so we still use it.
                    return trial_folder
                else:
                    trial = max_number + 1
            else:
                # Folder exists but contains data, so create a new one
                # with the next available numerical ID
                trial += 1
        
        # Create a new folder to save data to
        trial_folder = os.path.join(logs_folder, 'trial_' + str(trial))   
        os.makedirs(trial_folder)
        return trial_folder
    
    def check_goal(self, label):
        # if self.goal_synonyms == None:
        #     for i in range(self.llm_max_query):
        #         if self.goal_synonyms == None:
        #             goal_candidate = self.llm.check_goal(self.goal)
        #             try:
        #                 goal_candidate_lst = json.loads(goal_candidate)
        #                 if isinstance(goal_candidate_lst, list) and self.goal in goal_candidate_lst:
        #                     self.goal_synonyms = [i.lower() for i in goal_candidate_lst]
        #                     break
        #             except:
        #                 print('ERROR: No valid goal')
        if self.goal_synonyms == None:
            self.goal_synonyms = [self.goal]
        
        for i in self.goal_synonyms:
            if i in label:
                return True
        return False 

    def loop(self, obs, run_planning=True):
        print("Goal:", self.goal, "| Path", self.tmp_path)

        # Mapping
        print(">>> Mapping")
        parsed = self.mapper.parseImage(obs)
        prev_state = None if len(self.tmp_path) == 0 else self.tmp_path[-1]
        state = self.mapper.estimateState(prev_state, parsed)
        state, object_nodes = self.mapper.updateOSG(state, self.tmp_path, parsed)
        self.mapper.visualiseOSG(state)
        print(self.mapper.OSG.printGraph(output_json=False))

        if len(self.tmp_path) == 0 or state != self.tmp_path[-1]:
            print(f"Adding current state {state} to path as Place!")
            self.tmp_path.append(state)
        
        # Planning (or manual subgoal selection)
        if run_planning:
            # Use LLM-based Reasoner for planning
            next_subgoal_key, prev_subgoal_key = self.reasoner.generateExplorationPlan(
                self.goal, state, self.mapper.OSG
            )
            print(next_subgoal_key)
            next_subgoal_node = self.mapper.OSG.getNode(next_subgoal_key)
            in_view = next_subgoal_node["in_view"]
            if in_view is not None:
                view, bbox_id = in_view
            else:
                return None, None

        else:
            # Allow user to manually select subgoal
            while True:
                try:
                    selected_goal = input('Please provide the selected goal as <view> <bbox_id>, e.g. forward 2:\n')
                    print(f"Got: {selected_goal}")
                    if selected_goal.strip().lower() == "q":
                        import sys; sys.exit(0)
                    
                    view, bbox_id = selected_goal.split(' ')
                    bbox_id = int(bbox_id)
                    valid = (view in object_nodes) and len(object_nodes[view]) > bbox_id
                    if valid:
                        break
                except:
                    pass

        selected_node, selected_bbox = object_nodes[view][bbox_id]
        if self.mapper.OSG.isConnector(selected_node):
            print(f"Adding {selected_node} to path as Connector!")
            self.tmp_path.append(selected_node) # Hack to add connectors
        selected_bbox = tuple(map(lambda x: int(np.floor(x)), selected_bbox))
        print(selected_node, selected_bbox, view)
        print(self.mapper.OSG.spec.getClassSpec(selected_node.node_cls))
        print(">>>>>")
        self.last_subgoal = self.tmp_path
        return selected_bbox, view

    def test_loc(self, obs1, obs2):
        self.mapper.reset()
        parsed1 = self.mapper.parseImage(obs1)
        state = self.mapper.estimateState(None, parsed1)
        state, _ = self.mapper.updateOSG(state, [], parsed1)

        # Test new observation against scene graph
        parsed2 = self.mapper.parseImage(obs2)
        result = self.mapper.estimateState(state, parsed2)
        return not isinstance(result, tuple)