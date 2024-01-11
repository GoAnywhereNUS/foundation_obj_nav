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

from scene_graph import SceneGraph, default_scene_graph_specs
from model_interfaces import GPTInterface, VLM_BLIP, VLM_GroundingDino
import json, random

import time

def most_common(lst):
    if len(lst) > 0: 
        chosen = max(set(lst), key=lst.count)
    else:
        chosen = None
    return chosen

class Navigator:
    def __init__(
        self,
        scene_graph_specs=default_scene_graph_specs,
        llm_config_path="configs/gpt_config.yaml"
    ):
        # Set up foundation models for perception and reasoning
        self.llm = GPTInterface(config_path=llm_config_path)
        self.perception = {
            "object": VLM_GroundingDino(),
            "vqa": VLM_BLIP()
        }

        # Set up scene graph and state 
        if scene_graph_specs is None:
            # TODO: Query LLM to get the specs
            raise NotImplementedError
        self.scene_graph_specs = scene_graph_specs
        self.scene_graph = SceneGraph(scene_graph_specs)
        scene_graph_specs_dict = json.loads(scene_graph_specs)
        self.state_spec = scene_graph_specs_dict["state"]
        self.current_state = {node_type: None for node_type in self.state_spec}

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
        self.visualisation = True
        self.is_navigating = False
        self.success_flag = False
        self.GT = False
        self.semantic_annotations = None

        self.trial_folder = self.create_log_folder()
        self.action_log_path = os.path.join(self.trial_folder, 'action.txt')
        self.action_logging = open(
            self.action_log_path, 'a'
        )
        self.llm.reset(self.trial_folder)

        # Note: Env and controller to be implemented in subclass

        # TODO: Review these variables and parameters.
        self.defined_entrance = ['door', 'doorway', 'doorframe', 'window']


    def reset(self):
        # Reset scene graph
        self.scene_graph = SceneGraph(self.scene_graph_specs)

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

        # Reset logging
        self.trial_folder = self.create_log_folder()
        self.action_log_path = os.path.join(self.trial_folder, 'action.txt')
        self.action_logging = open(
            self.action_log_path, 'a'
        )
        # Reset LLM
        self.llm.reset(self.trial_folder)

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
            try:
                color = self.query_vqa(obj_img, f"What color is the {obj_name.split('_')[0]}?")
                material = self.query_vqa(obj_img, f"What material is the {obj_name.split('_')[0]}?")
            except:
                print(f'[Error] Cannot load image for {obj_name}')
                color = ''
                material = ''
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
        raise NotImplementedError

    def check_goal(self, label):
        if self.goal == 'sofa':
            if 'couch' in label:
                return True
            else:
                return self.goal in label
        if self.goal == 'bed':
            return self.goal == label
        if self.goal == 'toilet':
            if 'toilet' in label and 'seat' in label:
                return True
            elif 'toilet' in label and 'bowl' in label:
                return True
            else:
                return self.goal == label
        if self.goal == 'plant':
            if label == 'ornamental plant':
                return False
            else:
                return self.goal in label
        if self.goal == 'tv':
            if 'television' in label:
                return True
            else:
                return self.goal in label
        else:
            return self.goal in label

    def perceive(self, images):
        image_locations = {}
        image_objects = {}

        for view in images.keys():
            label = view.split('_')[0]
            image = images[label + '_rgb']
            image = Image.fromarray(image)
            location = self.query_vqa(image, "Which room is the photo?")
            image_locations[label] = (
                location.replace(" ", "") #clean space between "living room"
            )

            if self.GT: # Load Ground Truth Object Detection
                bbox_lst = []
                objlabel_lst = []
                cropped_img_lst = []
                semantic_gt = images[label +'_semantic']
                # Ignore small objects
                for instance in np.unique(semantic_gt):
                    instance_label = self.semantic_annotations[instance]
                    if instance_label not in ['wall', 'ceiling', 'floor', 'unknown']:
                        instance_index = np.argwhere(semantic_gt == instance)
                        min_y = np.min(instance_index[:,0])
                        max_y = np.max(instance_index[:,0])
                        min_x = np.min(instance_index[:,1])
                        max_x = np.max(instance_index[:,1])
                        if (max_x - min_x) * (max_y - min_y) < 50:
                            continue
                        elif (max_x - min_x) * (max_y - min_y) < 200 and (not self.check_goal(instance_label)):
                            continue
                        elif (max_x - min_x) * (max_y - min_y) < 1000 and random.uniform(0,1) > 0.2 and (not self.check_goal(instance_label)):
                            continue
                        objlabel_lst.append(instance_label)
                        bbox_lst.append([min_x, min_y, max_x, max_y])
                        cropped_img_lst.append(image.crop(np.array([min_x,min_y,max_x,max_y])))
                objects = (torch.tensor(bbox_lst), objlabel_lst, cropped_img_lst)
            
            else:
                # Load VLM to detect objects
                modified_entrance = []
                if (self.query_vqa(image, "Is there a door in the photo?") == 'yes'):
                    modified_entrance += self.defined_entrance
                if (
                    self.goal is not None 
                    and (self.query_vqa(
                            image, f"Is there a {self.goal} in the photo?"
                        ) == 'yes'
                    )
                ):
                    modified_entrance += [self.goal]
                if len(modified_entrance) > 0 :
                    objects = self.query_objects(image,  modified_entrance)
                else:
                    objects = self.query_objects(image)
                remove_id = []
                for i, objlabel in enumerate(objects[1]):
                    if 'glass' in objlabel:
                        remove_id +=  self.get_nearby_bbox(objects[0][i], objects[0], overlap_threshold = 0.7, distance_threshold = 0)
                bbox_lst = []
                objlabel_lst = []
                cropped_img_lst = []
                for i, objlabel in enumerate(objects[1]):
                    if i not in remove_id:
                        bbox_lst.append(objects[0][i])
                        objlabel_lst.append(objlabel)
                        cropped_img_lst.append(objects[2][i])
                objects = (torch.stack(bbox_lst, 0), objlabel_lst, cropped_img_lst)
            image_objects[label] = objects

        # TODO: Implement some reasonable fusion across all images
        return {
            "location": image_locations,
            "object": image_objects
        }

    def perceive_location(self, images):
        image_locations = {}
        for label in ['left_rgb', 'forward_rgb', 'right_rgb', 'rear_rgb']:
            image = images[label]
            image = Image.fromarray(image)
            location = self.query_vqa(image, "Which room is the photo?")
            image_locations[label] = (
                location.replace(" ", "") #clean space between "living room"
            )
        obs_location = most_common([image_locations['forward_rgb'], image_locations['left_rgb'], image_locations['right_rgb'], image_locations['rear_rgb']])
        # TODO: Implement some reasonable fusion across all images
        return obs_location

    def generate_query(self, discript, goal, query_type):
        if query_type == 'plan':
            start_question = "You see the partial layout of the apartment:\n"
            end_question = f"\nQuestion: Your goal is to find a {goal}. If any of the rooms in the layout are likely to contain the target object, reply the most probable room name, not any door name. If all the room are not likely contain the target object, provide the door you would select for exploring a new room where the target object might be found. Follow my format to state reasoning and answer. Please only use one word in answer."
            explored_item_list = [x for x in self.explored_node if isinstance(x, str)]
            explored_query = "The following has been explored: " + "["+ ", ".join(list(set(explored_item_list))) + "]. Please dont reply explored place or object."
            whole_query = start_question + discript + end_question + explored_query
        elif query_type == 'classify':
            start_question = "There is a list:"
            end_question = "Please eliminate redundant strings in the element from the list and classify them into \"room\", \"entrance\", and \"object\" classes. Ignore floor, ceiling and wall. Keep the number in the name. \nAnswer:"
            whole_query = start_question + discript + end_question
        elif query_type == 'local':
            start_question = "There is a list:"
            end_question = f"Please select one object that is most likely located near a {goal}. Please only select one object in the list and use this element name in answer. Use the exact name in the list. Always follow the format: Answer: <your answer>."
            whole_query = start_question + discript + end_question
        elif query_type == 'state_estimation':
            discript1 = "Depiction1: On the left, there is " + ", ".join(discript[0]['left']) + ". On the right, there is " + ", ".join(discript[0]['right']) + ". In front of me, there is " + ", ".join(discript[0]['forward']) + ". Behind me, there is " + ", ".join(discript[0]['rear']) + '\n'
            discript2 = "Depiction2: On the left, there is " + ", ".join(discript[1]['left']) + ". On the right, there is " + ", ".join(discript[1]['right']) + ".In front of me, there is " + ", ".join(discript[1]['forward']) + ". Behind me, there is " + ", ".join(discript[1]['rear']) + '\n'
            question = "These are depictions of what I observe from two different vantage points. Please tell me if these two viewpoints correspond to the same room. It's important to note that the descriptions may originate from two positions within the room, each with a distinct angle. Therefore, the descriptions may pertain to the same room but not necessarily capture the same elements. Please be aware that my viewing angle varies, so it is not necessary for the elements to align in the same direction. As long as the relative positions between objects are accurate, it is considered acceptable. Please assess the arrangement of objects and identifiable features in the descriptions to determine whether these two positions are indeed in the same place. Provide a response of True or False, along with supporting reasons."
            whole_query = discript1 + discript2 + question
        elif query_type == 'node_feature':
            target_node = discript[0].split("_")[0]
            target_feature = discript[1]
            candidate_entrances = discript[2]
            candidate_entrances_feature = discript[3]
            discript1 = f"We want to find a {target_node} that is near" + ", ".join(target_feature)
            discript2 = " .Now we have seen the following object: "
            for i, name in enumerate(candidate_entrances):
                discript2 += f" {name} that is near " + ", ".join(candidate_entrances_feature[i]) +". "
            question = "Please select one object that is most likely to be the object I want to find. Please only select one object and use this element name in answer. Use the exact name in the given sentences. Always follow the format: Answer: <your answer>."
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
                similar_room_obj = self.scene_graph.get_related_codes(similar_room, 'contains')
                similar_room_obj_descript = self.query_detailed_descript(similar_room_obj)

                store_ans = []
                for i in range(self.llm_max_query):
                    whole_query = self.generate_query([room_descript, similar_room_description], None, 'state_estimation')
                    answer = self.llm.query_state_estimation(whole_query)
                    if 'true' in answer:
                        store_ans.append(1)
                    elif 'false'in answer:
                        store_ans.append(0)
                    if len(store_ans) >= self.llm_sampling_query:
                        break
        
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
                seperate_ans = self.llm.query_object_class(whole_query)

                room_idx = seperate_ans.index('room')
                entrance_idx = seperate_ans.index('entrance')
                object_idx = seperate_ans.index('object')
                
                room_lst = seperate_ans[room_idx+1:entrance_idx]
                entrance_lst = seperate_ans[entrance_idx+1:object_idx]
                object_lst = seperate_ans[object_idx+1:]

                format_test = entrance_lst + object_lst
                qualified_node = []
                for item in format_test:
                    if item != 'none':
                        obj_name = item.split('_')[0]
                        idx = int(item.split('_')[1])
                        qualified_node.append(item)
                if len(qualified_node) > 0:
                    break
                else:
                    attempts += 1
            except:
                attempts += 1
        # Estimate State
    
        cropped_img_lst = []
        cleand_sensor_dir = []
        cleaned_object_lst = []
        to_be_updated_nodes = []
        for obj in object_lst:
            try:
                bb_idx = int(obj.split('_')[1])
            except:
                continue
            if bb_idx < len(cropped_imgs): # in case LLM return index out of range
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
        print("State Estimation:", est_state)
        if est_state != None: # The current state is already in scene graph
            self.current_state = est_state
            print(f'Existing node: {self.current_state}')
            to_be_updated_nodes, to_be_updated_nodes_feat = self.scene_graph.update_node(self.current_state)
            self.explored_node.append(self.last_subgoal) #TODO: check when we add this
            # TODO: how we update explored node

        else: # Enter a new node
            new_node = self.scene_graph.add_node("room", obs_location, {"active": True, "image": np.random.rand(4, 4), "description": room_description})
            # If last goal is entrance, we connect last state and current state with this entrance.
            if self.last_subgoal != None and self.scene_graph.is_type(self.last_subgoal, 'object'):
                self.scene_graph.add_edge(self.current_state, new_node, "connects to")
            
            self.current_state = new_node
            
            if self.last_subgoal != None:
                # If last subgoal is entrance, after we pass through the entrance, we need to decide which entrance we passed through
                if self.scene_graph.is_type(self.last_subgoal, 'entrance'):
                    find_last_entrance = False
                    self.scene_graph.add_edge(self.current_state, self.last_subgoal, "connects to")
                    self.explored_node.append(self.last_subgoal)
                    for idx, item in enumerate(entrance_lst):
                        if '_' in item:
                            entrance_name = item.split('_')[0]
                            bb_idx = int(item.split('_')[1])
                            sensor_dir = idx_sensordirection[bb_idx]
                            #TODO: how to choose the last entrance image
                            if sensor_dir == 'rear':
                                entrance_lst[idx] =  'LAST' + entrance_lst[idx]
                                find_last_entrance = True
                                break
                    # TODO: How to select the door just passed by
                    trial_time = 0
                    while (not find_last_entrance) and (trial_time < 3):
                        idx = random.choice(range(len(entrance_lst)))
                        trial_time += 1
                        if '_' in entrance_lst[idx]:
                            entrance_lst[idx] = 'LAST' + entrance_lst[idx]
                            find_last_entrance = True
                    if not find_last_entrance:
                        self.scene_graph.nodes()[self.last_subgoal]['active'] = False
                # If last subgoal is room, we directly connects two room
                elif self.scene_graph.is_type(self.last_subgoal, 'room'):
                    self.scene_graph.add_edge(self.current_state, self.last_subgoal, "connects to")
                    self.explored_node.append(self.last_subgoal)
                else:
                    self.explored_node.append(self.last_subgoal)

            print(f'Add new node: {self.current_state}')
        
        self.action_logging.write(f'[State]: {self.current_state}\n')
        
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
                    temp_obj = self.scene_graph.add_node("object", obj_name, {"active": True, "image": cropped_imgs[bb_idx],"bbox": obj_bbox[bb_idx], "cam_uuid": sensor_dir})
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
                        self.scene_graph.nodes()[temp_entrance]['image'] = cropped_imgs[bb_idx]
                        self.scene_graph.nodes()[temp_entrance]['bbox'] = obj_bbox[bb_idx]
                        self.scene_graph.nodes()[temp_entrance]['cam_uuid'] = sensor_dir
                        self.scene_graph.nodes()[temp_entrance]['active'] = True
                    else:
                        temp_entrance = self.scene_graph.add_node("entrance", entrance_name, {"active": True,"image": cropped_imgs[bb_idx],"bbox": obj_bbox[bb_idx],"cam_uuid": sensor_dir})
                    self.scene_graph.add_edge(self.current_state, temp_entrance, "connects to")
                    bbox_in_specific_dir = np.where(np.array(idx_sensordirection) == sensor_dir)[0] # get all objects in the direction
                    nearby_bbox_idx = self.get_nearby_bbox(obj_bbox[bb_idx],obj_bbox[bbox_in_specific_dir,])
                    for idx in nearby_bbox_idx:
                        if idx in bbox_idx_to_obj_name.keys():
                            new_obj = bbox_idx_to_obj_name[idx] 
                            self.scene_graph.add_edge(temp_entrance, new_obj, "is near")
                except:
                    print('ERROR')

        # Only when we are still in the same node, we need to update the features of doors.
        if len(to_be_updated_nodes) > 0:
            current_entrance = self.scene_graph.get_related_codes(self.current_state,'connects to')
            if len(current_entrance) > 0:
                print(' *** UPDATING ENTEANCE ***')
                current_entrance_feature = [self.scene_graph.get_related_codes(node,'is near') for node in current_entrance]
                for i, target_node in enumerate(to_be_updated_nodes):
                    target_features = to_be_updated_nodes_feat[i]
                    whole_query = self.generate_query([target_node, target_features, current_entrance, current_entrance_feature], None, 'node_feature')
                    store_ans = []
                    for i in range(self.llm_max_query):
                        seperate_ans = self.llm.query_node_feature(whole_query)
                        if len(seperate_ans) > 0 and seperate_ans[0] in current_entrance:
                            store_ans.append(seperate_ans[0])
                        if len(store_ans)  >= self.llm_sampling_query:
                            break
                    goal_node_name = most_common(store_ans)
                    if goal_node_name in current_entrance: # If we find a current door is similar to the important doors. Otherwise, we ignore it.
                        self.explored_node.append(goal_node_name)
                        self.scene_graph.combine_node(target_node, goal_node_name)
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
            if self.check_goal(obj):
                self.path = [obj]
                return self.path

        Scene_Discript = self.scene_graph.print_scene_graph(selected_node = self.current_state, pretty=False, skip_object=True)
        whole_query = self.generate_query(Scene_Discript, goal, 'plan')

        for i in range(self.llm_max_query):
            seperate_ans = self.llm.query(whole_query)
            if len(seperate_ans) > 0 and seperate_ans[0] in self.scene_graph.nodes():
                store_ans.append(seperate_ans[0])
            if len(store_ans)  >= self.llm_sampling_query:
                break
        
        # use whole lopp to choose the most common goal name that is in the scene graph
        print('[PLAN INFO] Receving Ans from LLM:', store_ans)

        store_ans_copy = store_ans.copy()
        goal_node_name = most_common(store_ans)

        # If no valid goal node, we explore the current state by default.
        if goal_node_name == None:
            goal_node_name = self.current_state

        print(f'[PLAN INFO] current state:{self.current_state}, goal state:{goal_node_name}')

        ########### End Query LLM for Plan #################

        try:
            if self.scene_graph.has_path(self.current_state, goal_node_name):
                path = self.scene_graph.plan_shortest_paths(self.current_state, goal_node_name)
            else:
                path = [self.current_state]
        except:
            path = [self.current_state]
            self.action_logging.write(f'[ERROR] Cannot Find a path between {self.current_state} and {goal_node_name}')

        # If we are already in the target room, Start local exploration in the room
        if path[-1] == self.current_state:
            self.explored_node.append(self.current_state)
            obj_lst = self.scene_graph.get_related_codes(self.current_state, 'contains', active_flag = True)
            cleaned_obj_lst = obj_lst.copy()
            for obj in obj_lst:
                if obj in self.explored_node:
                    obj_name = obj[:obj.index('_')]
                    cleaned_obj_lst = [element for element in cleaned_obj_lst if not element.startswith(obj_name)]
            sg_obj_Discript = "["+ ", ".join(cleaned_obj_lst) + "]"
            whole_query = self.generate_query(sg_obj_Discript, goal, 'local')
            
            store_ans = []
            nodes_in_view = self.scene_graph.nodes(active_flag=True)

            for i in range(self.llm_max_query):
                seperate_ans = self.llm.query_local_explore(whole_query)
                print(seperate_ans)
                if len(seperate_ans) > 0:
                    if seperate_ans[0] in nodes_in_view:
                        store_ans.append(seperate_ans[0])
                    elif seperate_ans[-1] in nodes_in_view:
                         store_ans.append(seperate_ans[-1])
                if len(store_ans)  >= self.llm_sampling_query:
                    break
            
            goal_node_name = most_common(store_ans)
            if goal_node_name == None:
                goal_node_name = random.choice(obj_lst)
            
            path = [goal_node_name]
        
        # if next subgoal is entrance, we mark it.
        if len(path) > 1:
            self.last_subgoal = path[1]
        else:
            self.last_subgoal = path[0]
        self.path = path
        
        if self.scene_graph.is_type(self.last_subgoal, 'room'): #TODO: add local exploration here
            # self.last_subgoal = random.choice(self.scene_graph.get_related_codes(self.last_subgoal, 'contains'))

            obj_lst = self.scene_graph.get_related_codes(self.current_state, 'contains')
            sg_obj_Discript = "["+ ", ".join(obj_lst) + "]"
            whole_query = self.generate_query(sg_obj_Discript, self.last_subgoal, 'local')
            
            store_ans = []
            nodes_in_view = self.scene_graph.nodes(active_flag=True)

            for i in range(self.llm_max_query):
                seperate_ans = self.llm.query_local_explore(whole_query)
                print(seperate_ans)
                if len(seperate_ans) > 0:
                    if seperate_ans[0] in nodes_in_view:
                        store_ans.append(seperate_ans[0])
                    elif seperate_ans[-1] in nodes_in_view:
                         store_ans.append(seperate_ans[-1])
                if len(store_ans)  >= self.llm_sampling_query:
                    break

            goal_node_name = most_common(store_ans)
            if goal_node_name == None:
                goal_node_name = random.choice(obj_lst)

            self.last_subgoal = goal_node_name
            path = [goal_node_name]
            self.path = path
        
        self.action_logging.write(f'[PLAN INFO] Path:{self.path}\n')
        self.action_logging.write(f'[Last Subgoal] Path:{self.last_subgoal}\n')

        if len(self.explored_node) > 2:
            if self.scene_graph.is_type(self.last_subgoal, 'object'):
                all_obj = self.scene_graph.get_related_codes(self.current_state, 'contains', active_flag = False)
                active_obj = self.scene_graph.get_related_codes(self.current_state, 'contains', active_flag = True)
                if (self.last_subgoal in all_obj) and (self.explored_node[-1] in all_obj) and (self.explored_node[-2] in all_obj):
                    if (self.explored_node[-1].split('_')[0] == self.last_subgoal.split('_')[0]) and (self.explored_node[-2].split('_')[0] == self.last_subgoal.split('_')[0]):
                        self.last_subgoal = random.choice(active_obj)
                        self.action_logging.write(f'[Error in Explored Node]: objects explored, random choose {self.last_subgoal}\n')
            elif self.scene_graph.is_type(self.last_subgoal, 'entrance'):
                all_entr = self.scene_graph.get_related_codes(self.current_state, 'connects to', active_flag = False)
                active_entr = self.scene_graph.get_related_codes(self.current_state, 'connects to', active_flag = True)
                if (self.last_subgoal in active_entr) and (self.explored_node[-1] in all_entr) and (self.explored_node[-2] in all_entr):
                    if (self.explored_node[-1].split('_')[0] == self.last_subgoal.split('_')[0]) and (self.explored_node[-2].split('_')[0] == self.last_subgoal.split('_')[0]):
                        if len(active_entr) <= 1:
                            self.last_subgoal = random.choice(self.scene_graph.get_related_codes(self.current_state, 'contains'))
                            self.action_logging.write(f'[Error in Explored Node]: Similar entrance explored, no more entrance, random choose {self.last_subgoal}\n')
                        else:
                            clean_active_entr = [ i for i in active_entr if i != self.last_subgoal]
                            self.last_subgoal = random.choice(clean_active_entr)
                            self.action_logging.write(f'[Error in Explored Node]: Similar entrance explored, so select another entrance, random choose {self.last_subgoal}\n')

        if self.scene_graph.nodes()[self.last_subgoal]['active'] == False:
            self.action_logging.write(f'[Error in Explored Node]: Inactive node {self.last_subgoal}\n')
            if self.scene_graph.is_type(self.last_subgoal, 'entrance'):
                active_entr = self.scene_graph.get_related_codes(self.current_state, 'connects to', active_flag = True)
                if len(active_entr) > 0 :
                    self.last_subgoal = random.choice(active_entr)
                    self.action_logging.write(f'[Error in Explored Node]: select active entrance, random choose {self.last_subgoal}\n')
            # if after update entrance, still cannot fing the valid entrance.
            if self.scene_graph.nodes()[self.last_subgoal]['active'] == False:
                active_obj = self.scene_graph.get_related_codes(self.current_state, 'contains', active_flag = True)
                if len(active_obj) > 0:
                    self.last_subgoal = random.choice(active_obj)
                    self.action_logging.write(f'[Error in Explored Node]: select active object, random choose {self.last_subgoal}\n')
        print(f'[Last Subgoal] Path:{self.last_subgoal}')
        self.action_logging.write(f'[Last Subgoal (Active)] Path:{self.last_subgoal}\n')
        self.action_logging.write(f'[Explored Node]:{self.explored_node}\n')

        return path

    def visualise_objects(self, obs, img_lang_obs):

        location = img_lang_obs['location']
        obj_lst = img_lang_obs['object']
    
        print(f'Location: {location}\nObjecet: {obj_lst}')
        obj_label_tmp = ['forward'] + img_lang_obs['object']['forward'][1] + ['left'] + img_lang_obs['object']['left'][1] + ['right'] + img_lang_obs['object']['right'][1] + ['rear'] + img_lang_obs['object']['rear'][1]
        self.action_logging.write(f'[Obs]: Location: {location}\nObjecet: {obj_label_tmp}\n')
                
        for direction in ['forward', 'left', 'right', 'rear']:
            obs_rgb = obs[direction + '_rgb']
            plt.imshow(obs_rgb.squeeze())
            ax = plt.gca()
            for i in range(len(img_lang_obs['object'][direction][1])):
                label = img_lang_obs['object'][direction][1][i]
                min_x, min_y, max_x, max_y = img_lang_obs['object'][direction][0][i]
                if self.check_goal(label):
                    ax.add_patch(plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, edgecolor='red', facecolor=(0,0,0,0), lw=2))
                else:
                    ax.add_patch(plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, edgecolor='green', facecolor=(0,0,0,0), lw=2))
                ax.text(min_x, min_y, label)
            plt.savefig(self.trial_folder + '/'  +  time.strftime("%Y%m%d%H%M%S") + "_" + direction + ".png")
            plt.clf()

    def check_current_obs(self, obs, img_lang_obs):
        for direction in ['forward', 'left', 'right', 'rear']:
            for i in range(len(img_lang_obs['object'][direction][1])):
                label = img_lang_obs['object'][direction][1][i]
                if self.check_goal(label):
                    next_position = img_lang_obs['object'][direction][0][i].type(torch.int64).tolist()
                    cam_uuid = direction +'_depth'
                    self.is_navigating = True
                    self.last_subgoal = self.goal
                    return True, next_position, cam_uuid
        return False, None, None
    
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

    def ground_plan_to_bbox(self):
        next_goal = self.last_subgoal
        print('Next Subgoal:', next_goal, self.scene_graph.scene_graph[next_goal])
        next_position = self.scene_graph.get_node_attr(next_goal)['bbox'].type(torch.int64).tolist()
        cam_uuid = self.scene_graph.get_node_attr(next_goal)['cam_uuid']+'_depth'
        return next_goal, next_position, cam_uuid

    def loop(self, obs):
        """
        Single iteration of high-level perception-reasoning loop for navigation.

        Returns:
            None
        """
        self.llm_loop_iter += 1

        # Observe
        self.action_logging.write(f'--------- Loop {self.llm_loop_iter} -----------\n')
        print('------------  Receive Lang Obs   -------------')
        img_lang_obs = self.perceive(obs)

        if self.visualisation:
            self.visualise_objects(obs, img_lang_obs)

        find_goal_flag, potenrial_next_pos, potenrial_cam_uuid = self.check_current_obs(obs, img_lang_obs)
        if find_goal_flag:
            return potenrial_next_pos, potenrial_cam_uuid

        # Update
        self.update_scene_graph(img_lang_obs)
        print('------------  Update Scene Graph   -------------')
        scene_graph_str = self.scene_graph.print_scene_graph(pretty=False,skip_object=False)
        self.action_logging.write(f'Scene Graph: {scene_graph_str}\n')
        print(scene_graph_str)

        # Plan
        print('-------------  Plan Path --------------')
        path = self.plan_path(self.goal)
        self.is_navigating = True

        next_goal, next_position, cam_uuid = self.ground_plan_to_bbox()
        self.action_logging.write(f'Path: {path}, Next Goal: {next_goal}\n')

        if self.visualisation:
            min_x, min_y, max_x, max_y = next_position

            plt.imshow(obs[cam_uuid[:-5]+'rgb'].squeeze())
            ax = plt.gca()
            ax.add_patch(plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, edgecolor='green', facecolor=(0,0,0,0), lw=2))
            ax.text(min_x, min_y, next_goal)
            plt.savefig(self.trial_folder + '/'  + time.strftime("%Y%m%d%H%M%S") + "_chosen_rgb_" + str(cam_uuid[:-6]) + ".png")
            plt.clf()

            plt.imshow(obs[cam_uuid[:-5]+'depth'].squeeze())
            ax = plt.gca()
            ax.add_patch(plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, edgecolor='green', facecolor=(0,0,0,0), lw=2))
            ax.text(min_x, min_y, next_goal)
            plt.savefig(self.trial_folder + '/'  + time.strftime("%Y%m%d%H%M%S") + "_chosen_depth_" + str(cam_uuid[:-6]) + ".png")
            plt.clf()


        if len(path) > 1:
            next_goal = path[1]
        else:
            next_goal = path[0]
    
        if len(path) > 1:
            next_goal = path[1]
        else:
            next_goal = path[0]
        return next_position, cam_uuid 

    def run(self):
        """
        Executes the navigation loop. To be implemented in each
        Navigator subclass.

        Returns:
            None
        """
        raise NotImplementedError
