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
from controller import *

import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def most_common(lst):
    if len(lst) > 0: 
        chosen = max(set(lst), key=lst.count)
    else:
        chosen = None
    return chosen

class NavigatorSimulation(Navigator):
    def __init__(
        self,
        scene_graph_specs=default_scene_graph_specs,
        llm_config_path="configs/gpt_config.yaml"
    ):
        
        super().__init__(
            scene_graph_specs=scene_graph_specs, 
            llm_config_path=llm_config_path
        )

        self.scene_graph_specs = scene_graph_specs
        self.llm_query_trial = 5
        self.defined_entrance = ['doorway', 'entrance', 'door frame']
        self.explored_node = []
        self.history = []
        self.path = [] # planned path in scene graph
        self.semantic_annotations = []
        self.GT = True
        self.last_subgoal = None

        # Setup simulator
        config = setup_env_config(default_config_path='configs/objectnav_hm3d_v2_with_semantic.yaml')
        self.config = config
        self.env = ObjNavEnv(habitat.Env(config=config), config)
        obs = self.env.reset()

        
    def reset(self):
        self.explored_node = []
        self.history = []
        self.path = [] # planned path in scene graph
        self.semantic_annotations = []
        obs = self.env.reset()
        self.scene_graph = SceneGraph(self.scene_graph_specs)
        self.llm.reset()
        self.last_subgoal = None

    def query_objects(self, image, suggested_objects=None):
        if suggested_objects is not None:
            return self.perception['object'].detect_specific_objects(image, suggested_objects)
        else:
            return self.perception['object'].detect_all_objects(image)

    def calculate_iou(self, bbox1, bbox2):
        # Calculate the intersection coordinates
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])

        # Calculate the area of intersection
        intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

        # Calculate the areas of each bounding box
        area_bbox1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        area_bbox2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        # Calculate the Union area
        union_area = area_bbox1 + area_bbox2 - intersection_area

        if union_area <= 0:
            return 0  # No overlap, IoU is 0

        # Calculate the IoU
        iou = intersection_area / union_area

        return iou

    def get_nearby_bbox(self, target_bbox, all_bbox, distance_threshold = 100, overlap_threshold = 0.1):
        #Sample data: Bounding boxes as (x_min, y_min, x_max, y_max)

        # Calculate the center of the specific bounding box
        specific_center = ((target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2)

        # # Find nearby bounding boxes 
        nearby_bboxes_idx = []
        for i in range(len(all_bbox)):
            bbox = all_bbox[i]
            if list(bbox) == list(target_bbox):
                continue
            if np.sqrt((bbox[0] - specific_center[0])**2 + (bbox[1] - specific_center[1])**2) <= distance_threshold:
                # print(np.sqrt((bbox[0] - specific_center[0])**2 + (bbox[1] - specific_center[1])**2))
                nearby_bboxes_idx.append(i)
            elif self.calculate_iou(target_bbox, bbox) > overlap_threshold:
                nearby_bboxes_idx.append(i)
        return nearby_bboxes_idx

    def query_vqa(self, image, prompt):
        return self.perception['vqa'].query(image, prompt)

    def query_detailed_descript(self, obj_names_list, obj_cropped_img_lst = None):
        update_descript_list = []
        for i, obj_name in enumerate(obj_names_list):
            if obj_cropped_img_lst == None:
                obj_img = self.scene_graph.get_node_attr(obj_name)['image']
            else:
                obj_img = obj_cropped_img_lst[i]
            color = self.query_vqa(obj_img, f"What color is the {obj_name.split('_')[0]}?")
            material = self.query_vqa(obj_img, f"What material is the {obj_name.split('_')[0]}?")
            descript =  color + ' ' + material + ' '+ obj_name.split('_')[0]
            update_descript_list.append(descript)
        return update_descript_list

    def _observe(self):
        """
        Get observations from the environment (e.g. render images for
        an agent in sim, or take images at current location on real robot).
        To be overridden in subclass.

        Return:
            images: dict of images taken at current pose
        """
        obs = self.env.get_observation()
        return {
            'left': obs['left_rgb'],
            'forward': obs['forward_rgb'],
            'right': obs['right_rgb'],
            'rear': obs['rear_rgb'],
            'left_semantic': obs['left_semantic'],
            'forward_semantic': obs['forward_semantic'],
            'right_semantic': obs['right_semantic'],
            'rear_semantic': obs['rear_semantic'],
        }

    def perceive(self, images):
        image_locations = {}
        image_objects = {}
        for label in ['left', 'forward', 'right', 'rear']:
            image = images[label]
            image = Image.fromarray(image)
            location = self.query_vqa(image, "Which room is the photo?")
            image_locations[label] = (
                location.replace(" ", "") #clean space between "living room"
            )

            if self.GT == True:
                bbox_lst = []
                objlabel_lst = []
                cropped_img_lst = []
                semantic_gt = images[label +'_semantic']
                for instance in np.unique(semantic_gt):
                    instance_label = self.semantic_annotations[instance]
                    if instance_label not in ['wall', 'ceiling', 'floor', 'unknown', 'door']:
                        instance_index = np.argwhere(semantic_gt == instance)
                        min_y = np.min(instance_index[:,0])
                        max_y = np.max(instance_index[:,0])
                        min_x = np.min(instance_index[:,1])
                        max_x = np.max(instance_index[:,1])

                        if (max_x - min_x) * (max_y - min_y) < 1000 and random.uniform(0,1) > 0.2:
                            continue
                        objlabel_lst.append(instance_label)
                        bbox_lst.append([min_x, min_y, max_x, max_y])
                        cropped_img_lst.append(image.crop(np.array([min_x,min_y,max_x,max_y])))
                objects = (torch.tensor(bbox_lst), objlabel_lst, cropped_img_lst)
            else:
                if self.query_vqa(image, "Is there a door in the photo?") == 'yes':
                    objects = self.query_objects(image,  self.defined_entrance)
                else:
                    objects = self.query_objects(image)
            image_objects[label] = objects

        # TODO: Implement some reasonable fusion across all images
        return {
            "location": image_locations,
            "object": image_objects
        }

    def perceive_location(self, images):
        image_locations = {}
        for label in ['left', 'forward', 'right', 'rear']:
            image = images[label]
            image = Image.fromarray(image)
            location = self.query_vqa(image, "Which room is the photo?")
            image_locations[label] = (
                location.replace(" ", "") #clean space between "living room"
            )
        obs_location = most_common([image_locations['forward'], image_locations['left'], image_locations['right'], image_locations['rear']])
        # TODO: Implement some reasonable fusion across all images
        return obs_location

    def generate_query(self, discript, goal, query_type):
        if query_type == 'plan':
            start_question = "You see the partial layout of the apartment:\n"
            end_question = f"\nQuestion: Your goal is to find a {goal}. If any of the rooms in the layout are likely to contain the target object, reply the most probable room name, not any door name. If all the room are not likely contain the target object, provide the door you would select for exploring a new room where the target object might be found. Follow my format to state reasoning and sample answer. Please only use one word in sample answer."
            explored_item_list = [x for x in self.explored_node if isinstance(x, str)]
            explored_query = "The following has been explored: " + "["+ ", ".join(list(set(explored_item_list))) + "]. Please dont reply explored place or object."
            whole_query = start_question + discript + end_question + explored_query
        elif query_type == 'classify':
            start_question = "There is a list:"
            end_question = "Please eliminate redundant strings in the element from the list and classify them into \"room\", \"entrance\", and \"object\" classes. Ignore floor, ceiling and wall. Keep the number in the name. \nSample Answer:"
            whole_query = start_question + discript + end_question
        elif query_type == 'local':
            start_question = "There is a list:"
            end_question = f"Please select one object that is most likely located near a {goal}. Please only select one object in the list and use one word in sample answer."
            whole_query = start_question + discript + end_question
        elif query_type == 'state_estimation':
            discript1 = "Depiction1: On the left, there is " + ", ".join(discript[0]['left']) + ". On the right, there is " + ", ".join(discript[0]['right']) + ". On the forward, there is " + ", ".join(discript[0]['forward']) + ". On the rear, there is " + ", ".join(discript[0]['rear']) + '\n'
            discript2 = "Depiction2: On the left, there is " + ", ".join(discript[1]['left']) + ". On the right, there is " + ", ".join(discript[1]['right']) + ". On the forward, there is " + ", ".join(discript[1]['forward']) + ". On the rear, there is " + ", ".join(discript[1]['rear']) + '\n'
            question = "These are depictions of what I observe from two different vantage points. Please tell me if these two viewpoints correspond to the same room. It's important to note that the descriptions may originate from two positions within the room, each with a distinct angle. Therefore, the descriptions may pertain to the same room but not necessarily capture the same elements. Please be aware that my viewing angle varies, so it is not necessary for the elements to align in the same direction. As long as the relative positions between objects are accurate, it is considered acceptable. Please assess the arrangement of objects and identifiable features in the descriptions to determine whether these two positions are indeed in the same place. Provide a response of True or False, along with supporting reasons."
            whole_query = discript1 + discript2 + question
        return whole_query

    def estimate_state(self, obs):
        """
        Queries the LLM with the observations and scene graph to
        get our current state.

        Inpit:
            obs: { "location": location, # room type, eg: livingroom_1
                    "object": objects # object labels and bounding boxes
                    }
        Return:
            state: agent's estimated state as dict in the format
                   {'floor': (new_flag, semantic_label), 'room': (new_flag, semantic_label), ...}
                   where new_flag indicates whether a new node needs to be added to this
                   particular level, and semantic_label is the language label to that node 
                   from the VLM.
        """
        
        est_state = None

        obj_label_descript = self.query_detailed_descript(obs['cleaned_object'], obs['cleaned_object_cropped_img'])
        room_descript = {'left':[], 'right':[], 'forward':[], 'rear':[]}
        for i, label in enumerate(obj_label_descript):
            room_descript[obs['cleand_sensor_dir'][i]].append(label)

        # TODO: add weight on differentt direction based on object num in each direction
        obs_location = most_common([obs['location']['forward'], obs['location']['left'], obs['location']['right'], obs['location']['rear']])
        
        room_lst_scene_graph = self.scene_graph.get_secific_type_nodes('room')
        all_room = [room[:room.index('_')] for room in room_lst_scene_graph]
        # if current room is already in scene graph
        if obs_location in all_room:
            indices = [index for index, element in enumerate(all_room) if element == obs_location]
            for i in indices:
                similar_room = room_lst_scene_graph[i]
                similar_room_description = self.scene_graph.get_node_attr(similar_room)['description']
                similar_room_obj = self.scene_graph.get_obj_in_room(similar_room)
                similar_room_obj_descript = self.query_detailed_descript(similar_room_obj)
                #TODO: how to decide whether they are the same room? LLM query
                # overlap = [element for element in similar_room_obj_descript if element in obj_label_descript]

                # threshold = min(len(similar_room_obj_descript), len(obj_label_descript)) * 0.7
                # is_similar = len(overlap) >= threshold
                # print('Overlap', overlap, obj_label_descript, similar_room_obj_descript)
                
                store_ans = []
                for i in range(self.llm_query_trial):
                    whole_query = self.generate_query([room_descript, similar_room_description], None, 'state_estimation')
                    chat_completion = self.llm.query_state_estimation(whole_query)
                    complete_response = chat_completion.choices[0].message.content.lower()
                    sample_response = complete_response[complete_response.find('sample answer:'):]
                    seperate_ans = re.split('\n|; |, | |sample answer:', sample_response)
                    seperate_ans = [i.replace('.','') for i in seperate_ans if i != '']
                    if 'true' in seperate_ans[0]:
                        store_ans.append(1)
                    elif 'false'in seperate_ans[0]:
                        store_ans.append(0)
                print("State Estimation:", store_ans)

                is_similar = most_common(store_ans)
                if is_similar:
                    est_state = similar_room
                    break
        return est_state, room_descript
    
    def update_scene_graph(self, obs, flag = False):
        """
        Updates scene graph using localisation estimate from LLM, and
        observations from VLM.

        Notice: currently, not use est_state

        Return:
            state: agent's current state as dict, e.g. {'floor': xxx, 'room': xxx, ...}
        """

        # TODO: Need panaromic view to estimate state
        
        # Deal with multiple rgn sensors
        obj_label = obs['object']['forward'][1] + obs['object']['left'][1] + obs['object']['right'][1] + obs['object']['rear'][1]
        obj_bbox = torch.cat((obs['object']['forward'][0], obs['object']['left'][0], obs['object']['right'][0], obs['object']['rear'][0]), dim=0)
        cropped_imgs = obs['object']['forward'][2] + obs['object']['left'][2] + obs['object']['right'][2] + obs['object']['rear'][2]
        obs_location = most_common([obs['location']['forward'], obs['location']['left'], obs['location']['right'], obs['location']['rear']])
        idx_sensordirection = ['forward' for i in range(len(obs['object']['forward'][1]))] + ['left' for i in range(len(obs['object']['left'][1]))] + ['right' for i in range(len(obs['object']['right'][1]))] + ['rear' for i in range(len(obs['object']['rear'][1]))] 
        # Add bbox index into obj label
        obj_label = [f'{item}_{index}' for index, item in enumerate(obj_label)]
        obs_obj_discript = "["+ ", ".join(obj_label) + "]"
        whole_query = self.generate_query(obs_obj_discript, None, 'classify')

        attempts = 0

        while attempts < 5:
            try:
                # Query LLM to classify detected objects in 'room','entrance' and 'object' 
                chat_completion = self.llm.query_object_class(whole_query)
                complete_response = chat_completion.choices[0].message.content.lower()
                complete_response = complete_response.replace(" ", "")
                seperate_ans = re.split('\n|,|:', complete_response)
                seperate_ans = [i.replace('.','') for i in seperate_ans if i != '']   

                room_idx = seperate_ans.index('room')
                entrance_idx = seperate_ans.index('entrance')
                object_idx = seperate_ans.index('object')
                
                room_lst = seperate_ans[room_idx+1:entrance_idx]
                entrance_lst = seperate_ans[entrance_idx+1:object_idx]
                object_lst = seperate_ans[object_idx+1:]

                format_test = entrance_lst + object_lst
                for item in format_test:
                    if '_' in item:
                        obj_name = item.split('_')[0]
                        idx = int(item.split('_')[1])
                break
            except:
                attempts += 1
        # Estimate State
    
        cropped_img_lst = []
        cleand_sensor_dir = []
        cleaned_object_lst = []
        for obj in object_lst:
            try:
                bb_idx = int(obj.split('_')[1])
            except:
                continue
            cropped_img_lst.append(cropped_imgs[bb_idx])
            cleand_sensor_dir.append(idx_sensordirection[bb_idx])
            cleaned_object_lst.append(obj)
        obs['cleaned_object'] = cleaned_object_lst
        obs['cleaned_object_cropped_img'] = cropped_img_lst
        obs['cleand_sensor_dir'] = cleand_sensor_dir

        print('-------------  State Estimation --------------')
        est_state, room_description = self.estimate_state(obs)
        print('Room Description', room_description) 
        # Update Room Node
        if est_state != None:
            self.current_state = est_state
            print(f'Existing node: {self.current_state}')
        else:
            self.current_state = self.scene_graph.add_node("room", obs_location, {"image": np.random.rand(4, 4), "description": room_description})
            if self.last_subgoal != None and self.scene_graph.is_type(self.last_subgoal, 'entrance'):
                self.scene_graph.add_edge(self.current_state, self.last_subgoal, "connects to")
                self.explored_node.append(self.last_subgoal)
                for idx, item in enumerate(entrance_lst):
                    if item == 'none':
                        continue
                    if '_' in item:
                        entrance_name = item.split('_')[0]
                        bb_idx = int(item.split('_')[1])
                        sensor_dir = idx_sensordirection[bb_idx]
                        if sensor_dir == 'rear':
                            entrance_lst[idx] =  'LAST' + entrance_lst[idx]
                            break
            print(f'Add new node: {self.current_state}')

        room_lst_scene_graph = self.scene_graph.get_secific_type_nodes('room')
        all_room = [room[:room.index('_')] for room in room_lst_scene_graph]
        
        # TODO: If the reply does not have '_', update fails.
        bbox_idx_to_obj_name = {}
        for item in object_lst:
            if item == 'none':
                continue
            try:
                if '_' in item:
                    obj_name = item.split('_')[0]
                    if obj_name in all_room: # if the object name is also room name, skip it. otherwise the room name may point to object node
                        continue
                    bb_idx = int(item.split('_')[1])
                    sensor_dir = idx_sensordirection[bb_idx]
                    temp_obj = self.scene_graph.add_node("object", obj_name, {"image": cropped_imgs[bb_idx],"bbox": obj_bbox[bb_idx], "cam_uuid": sensor_dir})
                    self.scene_graph.add_edge(self.current_state, temp_obj, "contains")
                    bbox_idx_to_obj_name[bb_idx] = temp_obj
            except:
                print(f'Scene Graph: Fail to add object item {item}')

        for item in entrance_lst:
            if item == 'none':
                continue
            if '_' in item:
                entrance_name = item.split('_')[0]
                bb_idx = int(item.split('_')[1])
                
                if self.current_state[:-2] == entrance_name: # handle wrong entrance name, (To be deleted).
                    continue
                try:
                    sensor_dir = idx_sensordirection[bb_idx] # get the sensor direction for this entrance
                    if 'LAST' in item:
                        temp_entrance = self.last_subgoal
                        self.scene_graph.nodes()[temp_entrance]['cam_uuid'] = cropped_imgs[bb_idx]
                        self.scene_graph.nodes()[temp_entrance]['bbox'] = obj_bbox[bb_idx]
                        self.scene_graph.nodes()[temp_entrance]['cam_uuid'] = sensor_dir
                    else:
                        temp_entrance = self.scene_graph.add_node("entrance", entrance_name, {"image": cropped_imgs[bb_idx],"bbox": obj_bbox[bb_idx],"cam_uuid": sensor_dir})
                    self.scene_graph.add_edge(self.current_state, temp_entrance, "connects to")
                    bbox_in_specific_dir = np.where(np.array(idx_sensordirection) == sensor_dir)[0] # get all objects in the direction
                    nearby_bbox_idx = self.get_nearby_bbox(obj_bbox[bb_idx],obj_bbox[bbox_in_specific_dir,])
                    for idx in nearby_bbox_idx:
                        if idx in bbox_idx_to_obj_name.keys():
                            new_obj = bbox_idx_to_obj_name[idx] 
                            self.scene_graph.add_edge(temp_entrance, new_obj, "is near")
                except:
                    print('ERROR')
        return est_state


    def plan_path(self, goal):
        '''
        Plan based on Scene graph
        Input:
            goal: string, target object
        Return:
            plan: list of node name from current state to goal state;
                  if already in the goal room, return object list that is most likely near goal
        '''

        ########### Begin Query LLM for Plan #################
        print('goal', goal)
        store_ans = []

        obj_lst_scene_graph = self.scene_graph.get_secific_type_nodes('object')
        for obj in obj_lst_scene_graph:
            if goal in obj:
                self.path = [obj]
                return self.path

        Scene_Discript = self.scene_graph.print_scene_graph(pretty=False)
        whole_query = self.generate_query(Scene_Discript, goal, 'plan')

        # query LLm for llm_query_trial times and select most common answer
        for i in range(self.llm_query_trial):
            chat_completion = self.llm.query(whole_query)
            complete_response = chat_completion.choices[0].message.content.lower()
            sample_response = complete_response[complete_response.find('sample answer:'):]
            seperate_ans = re.split('\n|; |, | |sample answer:', sample_response)
            seperate_ans = [i.replace('.','') for i in seperate_ans if i != ''] # to make sink. to sink
            if len(seperate_ans) > 0:
                store_ans.append(seperate_ans[0])
        
        # use whole lopp to choose the most common goal name that is in the scene graph
        print('[PLAN INFO] Receving Ans from LLM:', store_ans)

        store_ans_copy = store_ans.copy()
        goal_node_name = most_common(store_ans)
        # TODO: If the ans is not any valid node in sene graph, error
        while goal_node_name not in self.scene_graph.nodes():
            store_ans = [i for i in store_ans if i != goal_node_name]
            if len(store_ans) == 0:
                print(f'PLAN: cannot find a valid goal node name. Answer Store: {store_ans_copy}')
                print('ERROR')
                break
            goal_node_name = most_common(store_ans)
        
        print(f'[PLAN INFO] current state:{self.current_state}, goal state:{goal_node_name}')

        ########### End Query LLM for Plan #################

        path = self.scene_graph.plan_shortest_paths(self.current_state, goal_node_name)

        # If we are already in the target room, Start local exploration in the room
        if path[-1] == self.current_state:
            self.explored_node.append(self.current_state)
            obj_lst = self.scene_graph.get_obj_in_room(self.current_state)
            sg_obj_Discript = "["+ ", ".join(obj_lst) + "]"
            whole_query = self.generate_query(sg_obj_Discript, goal, 'local')
            
            store_ans = []
            for i in range(self.llm_query_trial):
                chat_completion = self.llm.query_local_explore(whole_query)
                complete_response = chat_completion.choices[0].message.content.lower()
                sample_response = complete_response[complete_response.find('sample answer:'):]
                seperate_ans = re.split('\n|; |, | |sample answer:', sample_response)
                seperate_ans = [i.replace('.','') for i in seperate_ans if i != '']
                if '_' in seperate_ans[0]:
                    store_ans.append(seperate_ans[0]) # ans should be separate_ans[0]
            
            goal_node_name = most_common(store_ans)
            while goal_node_name not in self.scene_graph.nodes():
                store_ans = [i for i in store_ans if i != goal_node_name]
                if len(store_ans) == 0:
                    print(f'PLAN: cannot find a valid goal node name. Answer Store: {store_ans_copy}')
                    print('ERROR')
                    break
                goal_node_name = most_common(store_ans)
        
            path = [goal_node_name]
        
        # if next ubgoal is entrance, we mark it.
        if len(path) > 1:
            print('path', path)
            if self.scene_graph.is_type(path[1], 'entrance'):
                self.last_subgoal = path[1]
        self.path = path
            
        print(f'[PLAN INFO] Path:{self.path}')

        return path

    def ground_plan_to_bbox(self):
        if len(self.path) > 1:
            next_goal = self.path[1]
        else:
            next_goal = self.path[0]
        print('Next Subgoal:', next_goal, self.scene_graph.scene_graph[next_goal])
        next_position = self.scene_graph.get_node_attr(next_goal)['bbox'].type(torch.int64).tolist()
        cam_uuid = self.scene_graph.get_node_attr(next_goal)['cam_uuid']+'_depth'
        return next_goal, next_position, cam_uuid
    
    def has_reached_subgoal(self, state, subgoal):
        raise NotImplementedError
    
    def send_navigation_subgoal(self, subgoal):
        raise NotImplementedError

    def loop(self):
        """
        Single iteration of the navigation loop

        Returns:
            None
        """
        if self.is_navigating:
            return
        
        # Get "observations" from VLMs
        image_lang_obs = self.observe()

        # Localise and update scene graph
        state = self.estimate_state(self, image_lang_obs)
        state = self.update_scene_graph(state, image_lang_obs)
        self.current_state = state

        # If not currently executing a plan, get new plan from LLM.
        # If currently executing a plan, get next subgoal if we have
        # successfully executed current subgoal, else re-plan with LLM.
        if self.plan is None or not self.has_reached_subgoal(state, self.last_subgoal):
            self.plan = self.plan_path()
        next_subgoal = self.plan[-1]

        self.is_navigating = True
        self.send_navigation_subgoal(next_subgoal)
        self.plan.pop()

    def run(self):
        """
        Executes the navigation loop. To be implemented in each
        Navigator subclass.

        Returns:
            None
        """
        raise NotImplementedError


def create_log_folder(log_folder = 'logs'):
    logs_folder = 'logs'

    all_folders = [folder for folder in os.listdir(logs_folder) if os.path.isdir(os.path.join(logs_folder, folder))]

    # Filter folders that start with "trial_"
    trial_folders = [folder for folder in all_folders if folder.startswith("trial_")]

    # Extract the numbers and find the maximum
    numbers = [int(folder.split("_")[1]) for folder in trial_folders]
    max_number = max(numbers, default=0)
    trial = max_number + 1

    trial_folder = os.path.join(logs_folder, 'trial_' + str(trial))
    if not os.path.exists(trial_folder):
        os.makedirs(trial_folder)
    
    return trial_folder

if __name__ == "__main__":


    test_scene = ['00800-TEEsavR23oF', '00802-wcojb4TFT35', '00813-svBbv1Pavdk', '00814-p53SfW6mjZe', '00820-mL8ThkuaVTM', 
                  '00824-Dd4bFSTQ8gi', '00829-QaLdnwvtxbs', '00832-qyAac8rV8Zk', '00835-q3zU7Yy5E5s', '00839-zt1RVoi7PcG', 
                  '00843-DYehNKdT76V', '00848-ziup5kvtCCR', '00853-5cdEh9F2hJL', '00873-bxsVRursffK', '00876-mv2HUxq3B53', 
                  '00877-4ok3usBNeis', '00878-XB4GS9ShBRE', '00880-Nfvxx8J5NCo', '00890-6s7QHgap2fW', '00891-cvZr5TUy5C5']

    device = torch.device('cuda:0')

    trial_folder = create_log_folder()
    action_log_path = os.path.join(trial_folder, 'action.txt')
    nav = NavigatorSimulation()
    first_time = True
    test_history = []

    while True:
        if first_time == True:
            first_time = False
        else:
            trial_folder = create_log_folder()
            action_log_path = os.path.join(trial_folder, 'action.txt')
        # scene_episode_id = nav.env.env.current_episode.scene_id + nav.env.env.current_episode.episode_id
        
        while nav.env.env.current_episode.episode_id not in ['1'] or ((nav.env.env.current_episode.scene_id + nav.env.env.current_episode.episode_id) in test_history):
            try:
                print('RESET', nav.env.env.current_episode.episode_id,nav.env.env.current_episode.scene_id )
                nav.reset()
            except:
                sys.exit(0)
        
        # import pdb
        # pdb.set_trace()

        try:
            test_history.append( nav.env.env.current_episode.scene_id + nav.env.env.current_episode.episode_id )
            controller = FMMController(device, env_config=nav.config)

            env_semantic_names = [s.category.name().lower() for s in nav.env.env.sim.semantic_annotations().objects]
            env_semantic_names = ['sofa' if x == 'couch' else x for x in env_semantic_names]
            env_semantic_names = ['toilet' if x == 'toilet seat' else x for x in env_semantic_names]

            nav.semantic_annotations = env_semantic_names

            goal = nav.env.env.current_episode.object_category
            if goal not in env_semantic_names:
                if goal == 'tv_monitor':
                    goal = 'tv'

            # goal_candidate =['chair', 'couch', 'plant', 'bed', 'toilet', 'tv']
            # # goal = random.choice(goal_candidate)
            # goal = 'couch'
            # while goal not in env_semantic_names:
            #     goal = random.choice(goal_candidate)
            # print('Goal', goal)
            
            with open(action_log_path, 'a') as file:
                file.write(f'[EPISODE ID]: {nav.env.env.current_episode.episode_id}\n')
                file.write(f'[SCNEN ID]: {nav.env.env.current_episode.scene_id}\n')
                file.write(f'[GOAL]: {goal}\n')
                obs = nav.env.get_observation()
                x = obs['gps'][0]
                y = obs['gps'][1]
                z = math.degrees(obs['compass'])
                file.write(f'[Pos]: {nav.env.env.sim.agents[0].get_state().position} [Rotation]: {nav.env.env.sim.agents[0].get_state().rotation} \n')
            
            cv2.namedWindow("Images")
            auto = False
            updated = True
            cv2.waitKey(1)
            import time
            
            loop_iter = 0
            step = 0

            while True:
                obs = nav.env.get_observation()
                t1 = time.time()
                controller.update(obs)
                images = controller.visualise(obs)
                cv2.imshow("Images", images)
                images = nav._observe()     
                cv2.waitKey(1)
                print('Pos', nav.env.env.sim.agents[0].get_state().position)

                if auto == False:

                    if loop_iter > 30 or step > 600:
                        with open(action_log_path, 'a') as file:
                            file.write(f"[END]: FAIL\n")
                        break

                    loop_iter += 1
                    # Observe
                    with open(action_log_path, 'a') as file:
                        file.write(f'--------- Loop {loop_iter} -----------\n')
                    img_lang_obs = nav.perceive(images)
                    print('------------  Receive Lang Obs   -------------')
                    location = img_lang_obs['location']
                    obj_lst = img_lang_obs['object']
                
                    print(f'Location: {location}\nObjecet: {obj_lst}')
                    obj_label_tmp = ['forward'] + img_lang_obs['object']['forward'][1] + ['left'] + img_lang_obs['object']['left'][1] + ['right'] + img_lang_obs['object']['right'][1] + ['rear'] + img_lang_obs['object']['rear'][1]

                    with open(action_log_path, 'a') as file:
                        file.write(f'[Obs]: Location: {location}\nObjecet: {obj_label_tmp}\n')
                    for direction in ['forward', 'left', 'right', 'rear']:
                        obs_rgb = images[direction]
                        plt.imshow(obs_rgb.squeeze())
                        ax = plt.gca()
                        for i in range(len(img_lang_obs['object'][direction][1])):
                            label = img_lang_obs['object'][direction][1][i]
                            min_x, min_y, max_x, max_y = img_lang_obs['object'][direction][0][i]
                            ax.add_patch(plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, edgecolor='green', facecolor=(0,0,0,0), lw=2))
                            ax.text(min_x, min_y, label)
                        plt.savefig(trial_folder + '/'  +  time.strftime("%Y%m%d%H%M%S") + "_" + direction + ".png")
                        plt.clf()
                    # Update

                    nav.update_scene_graph(img_lang_obs)
                    print('------------  Update Scene Graph   -------------')
                    scene_graph_str = nav.scene_graph.print_scene_graph(pretty=False,skip_object=False)
                    print(scene_graph_str)

                    with open(action_log_path, 'a') as file:
                        file.write(f'Scene Graph: {scene_graph_str}\n')
                    # Plan

                    print('-------------  Plan Path --------------')
                    path = nav.plan_path(goal)
                    next_goal, next_position, cam_uuid = nav.ground_plan_to_bbox()
                    
                    print(f'Path: {path}\n Next Goal: {next_goal}')
                    with open(action_log_path, 'a') as file:
                        file.write(f'Path: {path}, Next Goal: {next_goal}\n')

                    min_x, min_y, max_x, max_y = next_position


                    plt.imshow(obs[cam_uuid[:-5]+'rgb'].squeeze())
                    ax = plt.gca()
                    ax.add_patch(plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, edgecolor='green', facecolor=(0,0,0,0), lw=2))
                    plt.savefig(trial_folder + '/'  + time.strftime("%Y%m%d%H%M") + "_chosen_rgb_" + str(cam_uuid[:-6]) + ".png")
                    plt.clf()

                    plt.imshow(obs[cam_uuid[:-5]+'depth'].squeeze())
                    ax = plt.gca()
                    ax.add_patch(plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, edgecolor='green', facecolor=(0,0,0,0), lw=2))
                    plt.savefig(trial_folder + '/'  + time.strftime("%Y%m%d%H%M") + "_chosen_depth_" + str(cam_uuid[:-6]) + ".png")
                    plt.clf()

                    controller.set_subgoal_image(next_position, cam_uuid, obs, get_camera_matrix(640, 480, 90))
                    auto = True
                # Action
                else:
                    action, stop = controller.step()
                    if stop or action == None:
                        auto = False
                        nav.explored_node.append(nav.last_subgoal)
                        img_lang_obs = nav.perceive(images)
                        location = img_lang_obs['location']
                        obj_lst = img_lang_obs['object']
                        obj_label_tmp = ['forward'] + img_lang_obs['object']['forward'][1] + ['left'] + img_lang_obs['object']['left'][1] + ['right'] + img_lang_obs['object']['right'][1] + ['rear'] + img_lang_obs['object']['rear'][1]
                        succeed_flag = False
                        with open(action_log_path, 'a') as file:
                            file.write(f"[Action]: Reach the point\n")
                            for obj in obj_label_tmp:
                                if goal in obj:
                                    file.write(f"[END]: SUCCESS\n")
                                    succeed_flag = True
                                    for direction in ['forward', 'left', 'right', 'rear']:
                                        obs_rgb = images[direction]
                                        plt.imshow(obs_rgb.squeeze())
                                        ax = plt.gca()
                                        for i in range(len(img_lang_obs['object'][direction][1])):
                                            label = img_lang_obs['object'][direction][1][i]
                                            min_x, min_y, max_x, max_y = img_lang_obs['object'][direction][0][i]
                                            if goal in label:
                                                ax.add_patch(plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, edgecolor='red', facecolor=(0,0,0,0), lw=2))
                                            else:
                                                ax.add_patch(plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, edgecolor='green', facecolor=(0,0,0,0), lw=2))
                                            ax.text(min_x, min_y, label)
                                        plt.savefig(trial_folder + '/'  +  time.strftime("%Y%m%d%H%M%S") + "_" + direction + "end.png")
                                        plt.clf()
                                    break
                        if succeed_flag:
                            break
                    else:
                        print("(Auto) Action:", action)
                        nav.env.act(action)
                        step += 1

                        if step > 600:
                            with open(action_log_path, 'a') as file:
                                file.write(f"[END]: FAIL\n")
                            break
                        current_loc = nav.perceive_location(images)
                        nav.history.append(current_loc)
                        print(f'Current Loc: {current_loc}') 
                        with open(action_log_path, 'a') as file:
                            file.write(f"[Action]: {action}\n")
                            file.write(f'[Pos]: {nav.env.env.sim.agents[0].get_state().position} [Rotation]: {nav.env.env.sim.agents[0].get_state().rotation} \n')
            cv2.destroyAllWindows()
        except:
            with open(action_log_path, 'a') as file:
                file.write(f"[END]: LLM QUERY ERROR\n")
            continue