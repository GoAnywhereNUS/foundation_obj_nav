import os
import cv2
import numpy as np
import re
from PIL import Image
import logging
import matplotlib.pyplot as plt
import torch

import habitat_sim

from navigator import *
from scene_graph import SceneGraph, default_scene_graph_specs
from utils.habitat_utils import setup_sim_config

import pdb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def most_common(lst):
    return max(set(lst), key=lst.count)

class NavigatorTeleport(Navigator):
    def __init__(
        self,
        scene_graph_specs=default_scene_graph_specs,
        llm_config_path="configs/gpt_config.yaml",
    ):
        
        super().__init__(
            scene_graph_specs=scene_graph_specs, 
            llm_config_path=llm_config_path
        )

        self.llm_query_trial = 3
        self.defined_entrance = ['door', 'doorway', 'entrance']
        self.explored_node = []

        # Setup simulator
        sim_config = setup_sim_config()
        self.sim = habitat_sim.Simulator(sim_config)
        self.sim_agent = self.sim.initialize_agent(0)

    def reset(self):
        self.scene_graph = SceneGraph(self.scene_graph_specs)
        self.llm.reset()

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
    
    def _observe(self):
        """
        Get observations from the environment (e.g. render images for
        an agent in sim, or take images at current location on real robot).
        To be overridden in subclass.

        Return:
            images: dict of images taken at current pose
        """
        obs = self.sim.get_sensor_observations()
        # print('original',obs['left_rgb'].shape)
        # print('channel 4',obs['left_rgb'][:,:,3])
        # print('change', cv2.cvtColor(obs['left_rgb'], cv2.COLOR_BGR2RGB).shape)
        # print(sum(sum(obs['left_rgb'][:,:,0] == cv2.cvtColor(obs['left_rgb'], cv2.COLOR_BGR2RGB)[:,:,0])))
        # print(sum(sum(obs['left_rgb'][:,:,1] == cv2.cvtColor(obs['left_rgb'], cv2.COLOR_BGR2RGB)[:,:,1])))
        # print(sum(sum(obs['left_rgb'][:,:,2] == cv2.cvtColor(obs['left_rgb'], cv2.COLOR_BGR2RGB)[:,:,2])))
        # return {
        #     'left': cv2.cvtColor(obs['left_rgb'], cv2.COLOR_BGR2RGB),
        #     'forward': cv2.cvtColor(obs['forward_rgb'], cv2.COLOR_BGR2RGB),
        #     'right': cv2.cvtColor(obs['right_rgb'], cv2.COLOR_BGR2RGB),
        #     'rear': cv2.cvtColor(obs['rear_rgb'], cv2.COLOR_BGR2RGB),
        # }
        return {
            'left': obs['left_rgb'],
            'forward': obs['forward_rgb'],
            'right': obs['right_rgb'],
            'rear': obs['rear_rgb'],
        }

    def perceive(self, images):
        image_locations = {}
        image_objects = {}
        for label, image in images.items():
            image = Image.fromarray(image)
            location = self.query_vqa(image, "Which room is the photo?")
            image_locations[label] = (
                location.replace(" ", "") #clean space between "living room"
            )
            # print(f'Entrance Check: {self.query_vqa(image, "Is there a door in the photo?"), self.query_vqa(image, "Is the door in the photo open?")}')
            
            if self.query_vqa(image, "Is there a door in the photo?") == 'yes':
                objects = self.query_objects(image,  self.defined_entrance)
                # print(f'Update Detection added door: {objects}')
            else:
                objects = self.query_objects(image)
            image_objects[label] = objects

        # TODO: Implement some reasonable fusion across all images
        print(image_locations, image_objects)
        return {
            "location": image_locations,
            "object": image_objects
        }
    
    def generate_query(self, discript, goal, query_type):
        if query_type == 'plan':
            start_question = "You see the partial layout of the apartment:\n"
            end_question = f"\nQuestion: Your goal is to find a {goal}. If any of the rooms in the layout are likely to contain the target object, reply the most probable room name, not any door name. If all the room are not likely contain the target object, provide the door you would select for exploring a new room where the target object might be found. Follow my format to state reasoning and sample answer."
            explored_query = "The following has been explored: " + "["+ ", ".join(self.explored_node) + "]. Please dont reply explored place or object."
            whole_query = start_question + discript + end_question + explored_query
        elif query_type == 'classify':
            start_question = "There is a list:"
            end_question = "Please eliminate redundant strings in the element from the list and classify them into \"room\", \"entrance\", and \"object\" classes.\nSample Answer:"
            whole_query = start_question + discript + end_question
        elif query_type == 'local':
            start_question = "There is a list:"
            end_question = f"Please select one object that is most likely located near a {goal}."
            whole_query = start_question + discript + end_question
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

        obj_label = [item.split('_')[0] for item in obs['cleaned_object'] ]
        # TODO: add weight on differentt direction based on object num in each direction
        obs_location = most_common([obs['location']['forward'], obs['location']['left'], obs['location']['right'], obs['location']['rear']])
        
        room_lst_scene_graph = self.scene_graph.get_secific_type_nodes('room')
        all_room = [room[:room.index('_')] for room in room_lst_scene_graph]
        # if current room is already in scene graph
        if obs_location in all_room:
            indices = [index for index, element in enumerate(all_room) if element == obs_location]
            for i in indices:
                # #TODO: now use code to calculate the overlap of objects in two room, maybe could use LLM for this. 
                similar_room = room_lst_scene_graph[i]
                similar_room_obj = [item.split('_')[0] for item in self.scene_graph.get_obj_in_room(similar_room)]
                similar_room_obj = list(set(similar_room_obj))
                obj_label = list(set(obj_label))
                overlap = [element for element in similar_room_obj if element in obj_label]
                threshold = min(len(similar_room_obj), len(obj_label)) * 0.7
                is_similar = len(overlap) >= threshold
                print('Overlap', overlap, obj_label, similar_room_obj)
                if is_similar:
                    est_state = similar_room
                    break
        return est_state
    
    def update_scene_graph(self, obs):
        """
        Updates scene graph using localisation estimate from LLM, and
        observations from VLM.

        Notice: currently, not use est_state

        Return:
            state: agent's current state as dict, e.g. {'floor': xxx, 'room': xxx, ...}
        """

        # TODO: Need panaromic view to estimate state
        
        # Begin Query LLM to classify detected objects in 'room','entrance' and 'object' 
        
        obj_label = obs['object']['forward'][1] + obs['object']['left'][1] + obs['object']['right'][1] + obs['object']['rear'][1]
        obj_bbox = torch.cat((obs['object']['forward'][0], obs['object']['left'][0], obs['object']['right'][0], obs['object']['rear'][0]), dim=0)
        obs_location = most_common([obs['location']['forward'], obs['location']['left'], obs['location']['right'], obs['location']['rear']])
        
        # Add bbox index into obj label
        obj_label = [f'{item}_{index}' for index, item in enumerate(obj_label)]
        obs_obj_discript = "["+ ", ".join(obj_label) + "]"
        whole_query = self.generate_query(obs_obj_discript, None, 'classify')

        
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

        # Estimate State
    
        obs['cleaned_object'] = object_lst

        print('-------------  State Estimation --------------')
        est_state = self.estimate_state(obs)

        # Update Room Node
        if est_state != None:
            self.current_state = est_state
            print(f'dExisting node: {self.current_state}')
        else:
            self.current_state = self.scene_graph.add_node("room", obs_location, {"image": np.random.rand(4, 4)})
            if self.last_subgoal != None and self.scene_graph.is_type(self.last_subgoal, 'entrance'):
                self.scene_graph.add_edge(self.current_state, self.last_subgoal, "connects to")
                self.explored_node.append(self.last_subgoal)
            print(f'Add new node: {self.current_state}')

        # TODO: If the replt does not have '_', update fails.
        bbox_idx_to_obj_name = {}
        for item in object_lst:
            if item == 'none':
                continue
            try:
                if '_' in item:
                    obj_name = item.split('_')[0]
                    bb_idx = int(item.split('_')[1])
                    temp_obj = self.scene_graph.add_node("object", obj_name, {"image": np.random.rand(4, 4)})
                    self.scene_graph.add_edge(self.current_state, temp_obj, "contains")
                    bbox_idx_to_obj_name[bb_idx] = temp_obj
            except:
                logging.warning(f'Scene Graph: Fail to add object item {item}')

        for item in entrance_lst:
            if item == 'none':
                continue
            if '_' in item:
                entrance_name = item.split('_')[0]
                bb_idx = int(item.split('_')[1])
                
                if self.current_state[:-2] == entrance_name:
                    continue
                temp_entrance = self.scene_graph.add_node("entrance", entrance_name, {"image": np.random.rand(4, 4)})
                self.scene_graph.add_edge(self.current_state, temp_entrance, "connects to")

                nearby_bbox_idx = self.get_nearby_bbox(obj_bbox[bb_idx],obj_bbox)
                for idx in nearby_bbox_idx:
                    if idx in bbox_idx_to_obj_name.keys():
                        new_obj = bbox_idx_to_obj_name[idx] 
                        self.scene_graph.add_edge(temp_entrance, new_obj, "is near")

        logging.info(f'Scene Graph: {self.scene_graph.print_scene_graph(pretty=False, skip_object=False)}')
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
                logging.error(f'PLAN: cannot find a valid goal node name. Answer Store: {store_ans_copy}')
                print(f'PLAN: cannot find a valid goal node name. Answer Store: {store_ans_copy}')
                raise NotImplementedError
            goal_node_name = most_common(store_ans)
        
        print(f'[PLAN INFO] current state:{self.current_state}, goal state:{goal_node_name}')

        ########### End Query LLM for Plan #################

        path = self.scene_graph.plan_shortest_paths(self.current_state, goal_node_name)

        # If we are already in the target room, Start local exploration in the room
        # TODO: maybe later implement to query llm for multiple times.
        if path[-1] == self.current_state:
            self.explored_node.append(self.current_state)
            obj_lst = self.scene_graph.get_obj_in_room(self.current_state)
            sg_obj_Discript = "["+ ", ".join(obj_lst) + "]"
            whole_query = self.generate_query(sg_obj_Discript, goal, 'local')

            chat_completion = self.llm.query_local_explore(whole_query)
            complete_response = chat_completion.choices[0].message.content.lower()
            # TODO: handle llm return multiple answers, it will use "sample answers:"
            sample_response = complete_response[complete_response.find('sample answer:'):]
            seperate_ans = re.split('\n|; |, | |sample answer:', sample_response)
            seperate_ans = [i.replace('.','') for i in seperate_ans if i != '']
            path = seperate_ans # ans should be separate_ans[0]
        
        # if next ubgoal is entrance, we mark it.
        if len(path) > 1:
            print('path', path)
            if self.scene_graph.is_type(path[1], 'entrance'):
                self.last_subgoal = path[1]
        logging.warning(f'Path: {path}')
        return path

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


if __name__ == "__main__":
    nav = NavigatorTeleport()
    goal = 'oven'

    cv2.namedWindow("Viz")
    img_lang_obs = None

    while True:
        images = nav._observe()
        display_img = cv2.cvtColor(images['forward'], cv2.COLOR_BGR2RGB) 
        cv2.imshow("Viz", display_img)        
        key = cv2.waitKey(0)
        if key == ord('a'):
            nav.sim_agent.act('turn_left')
        elif key == ord('d'):
            nav.sim_agent.act('turn_right')
        elif key == ord('w'):
            nav.sim_agent.act('move_forward')
        elif key == ord('q'):
            break
        elif key == ord('o'):
            img_lang_obs = nav.perceive(images)
            print('------------  Receive Lang Obs   -------------')
            print(img_lang_obs)
        elif key == ord('u'):
            if img_lang_obs == None:
                print('Current Obs is None')
            else:
                print('------------  Curernt Lang Obs   -------------')
                location = img_lang_obs['location']
                obj_lst = img_lang_obs['object']
                print(f'Location: {location}\nObjecet: {obj_lst}')
                nav.update_scene_graph(img_lang_obs)
                print('------------  Update Scene Graph   -------------')
                print(nav.scene_graph.print_scene_graph(pretty=False,skip_object=False))
        elif key == ord('i'):
            path = nav.plan_path(goal)
            print('-------------  Plan Path --------------')
            print(path)
        elif key == ord('s'):
            print('-------------  Save Obs --------------')
            Image.fromarray(images['forward']).save('/home/zhanxin/Desktop/SceneGraph/test.png')
        elif key == ord('n'):
            print('-------------  Contrl Success --------------')
            nav.explored_node.append(nav.last_subgoal)

    cv2.destroyAllWindows()