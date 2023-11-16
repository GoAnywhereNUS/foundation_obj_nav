import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import sys
import cv2
import git
import imageio
import magnum as mn
import numpy as np
import pdb
from matplotlib import pyplot as plt
import re
from PIL import Image


# Set up logging
import logging
logging.basicConfig(filename='log/llm.log', level=logging.INFO, format='%(asctime)s | %(levelname)s: %(message)s')

sys.path.append("/home/zhanxin/Desktop/habitat-sim")
import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

from navigator import *

from scene_graph import SceneGraph, default_scene_graph_specs
from model_interfaces import GPTInterface, VLM_BLIP, VLM_GroundingDino


# This is the scene we are going to load, support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
test_scene = "/home/zhanxin/Desktop/SceneGraph/data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.3,  # Height of sensors in meters, relative to the agent
    "width": 640,  # Spatial resolution of the observations
    "height": 480,
}

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
        # Initialise foundation models used for agent's functions
        '''
        self.llm = GPTInterface(config_path=llm_config_path)
        self.perception = {
            "object": VLM_GroundingDino(),
            "vqa": VLM_BLIP()
        }
        self.scene_graph_specs = scene_graph_specs
        self.scene_graph = SceneGraph(scene_graph_specs)
        self.state_spec = self.scene_graph_specs["state"]
        self.current_state = {node_type: None for node_type in self.state_spec}
        self.plan = None
        self.last_subgoal = None
        self.is_navigating = False
        self.goal = None
        '''

    def reset(self):
        self.scene_graph = SceneGraph(self.scene_graph_specs)
        self.llm.reset()

    def query_objects(self, image, suggested_objects=None):
        if suggested_objects is not None:
            return self.perception['object'].detect_specific_objects(image, suggested_objects)
        else:
            return self.perception['object'].detect_all_objects(image)

    def query_vqa(self, image, prompt):
        return self.perception['vqa'].query(image, prompt)
    
    def _get_image(self):
        """
        Get most recent image from embodied agent. 
        To be overridden in each Navigator subclass.

        Returns:
            image: PIL.Image object
        """
        raise NotImplementedError
    
    def observe(self, obs_rgb):
        """
        Generate observations.

        Return:
            location: name of current location answered by VQA
            objects: list of all objects observed in current vicinity
        """
        image = obs_rgb
        location = self.query_vqa(image, "Which room is the photo?")
        objects = self.query_objects(image)
        return {
            "location": location,
            "object": objects
        }
    
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
        raise NotImplementedError
    
    def update_scene_graph(self, est_state, obs):
        """
        Updates scene graph using localisation estimate from LLM, and
        observations from VLM.

        Notice: currently, not use est_state

        Return:
            state: agent's current state as dict, e.g. {'floor': xxx, 'room': xxx, ...}
        """

        # TODO: Need panaromic view to estimate state
        
        # Begin Query LLM to classify detected objects in 'room','entrance' and 'object' 
        start_question = "There is a list."
        obj_label = obs['object'][1]
        Obs_obj_Discript = "["+ ", ".join(obj_label) + "]"
        end_question = "Please eliminate redundant strings in the element from the list and classify them into \"room\", \"entrance\", and \"object\" classes.\nSample Answer:"
        whole_query = start_question + Obs_obj_Discript + end_question

        chat_completion = self.llm.query_object_class(whole_query)
        complete_response = chat_completion.choices[0].message.content.lower()
        complete_response = complete_response.replace(" ", "")
        seperate_ans = re.split('\n|,|:', complete_response)
        seperate_ans = [i for i in seperate_ans if i != '']
        
        class_flag = None
        

        # Update Room Node
        # TODO: whether we need to add new room node
        # Currently we assume the layout only has one room for each room type
        room_lst_scene_graph = self.scene_graph.get_secific_type_nodes('room')
        diff_room = [room[:room.index('_')] for room in room_lst_scene_graph]
        # if current room is already in scene graph
        if obs['location'] in diff_room:
            self.current_state = room_lst_scene_graph[diff_room.index(obs['location'])]

            if self.last_subgoal != None and self.scene_graph.is_type(self.last_subgoal, 'entrance'):
                self.scene_graph.add_edge(self.current_state, self.last_subgoal, "contains")
        else:
            self.current_state = self.scene_graph.add_node("room", obs['location'], {"image": np.random.rand(4, 4)})

        for item in seperate_ans:
            if item == 'none':
                continue
            class_mapping = {
                'room': 'ROOM',
                'entrance': 'ENTRANCE',
                'object': 'OBJECT',
            }
            class_flag = class_mapping.get(item, class_flag)
            # Update llm output flag
            if item in class_mapping.keys():
                continue
            # Update entrance Node
            if class_flag == 'ENTRANCE':
                if self.current_state[:-2] == item:
                    continue
                temp_entrance = self.scene_graph.add_node("entrance", item, {"image": np.random.rand(4, 4)})
                self.scene_graph.add_edge(self.current_state, temp_entrance, "contains")
                # TODO: Calculate nearby labels
            # Update object Node
            elif class_flag == 'OBJECT':
                try:
                    temp_obj = self.scene_graph.add_node("object", item, {"image": np.random.rand(4, 4)})
                    self.scene_graph.add_edge(self.current_state, temp_obj, "contains")
                except:
                    logging.warning(f'Scene Graph: Fail to add object item {item}')

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

        store_ans = []
        start_question = "You see the partial layout of the apartment:\n"
        Scene_Discript = self.scene_graph.print_scene_graph(pretty=False)
        end_question = f"\nQuestion: Your goal is to find a {goal}. If any of the rooms in the layout are likely to contain the target object, reply the most probable room name, not any door name. If all the room are not likely contain the target object, provide the door you would select for exploring a new room where the target object might be found."
        whole_query = start_question + Scene_Discript + end_question

        # query LLm for llm_query_trial times and select most common answer
        for i in range(self.llm_query_trial):
            chat_completion = self.llm.query(whole_query)
            complete_response = chat_completion.choices[0].message.content.lower()
            sample_response = complete_response[complete_response.find('sample answer:'):]
            seperate_ans = re.split('\n|; |, | |sample answer:', sample_response)
            seperate_ans = [i for i in seperate_ans if i != '']
            if len(seperate_ans) > 0:
                store_ans.append(seperate_ans[0])
        
        # use whole lopp to choose the most common goal name that is in the scene graph
        store_ans_copy = store_ans.copy()
        goal_node_name = most_common(store_ans)
        while goal_node_name not in self.scene_graph.nodes():
            store_ans = [i for i in seperate_ans if i != goal_node_name]
            if len(store_ans) == 0:
                logging.error(f'PLAN: cannot find a valid goal node name. Answer Store: {store_ans_copy}')
                print(f'PLAN: cannot find a valid goal node name. Answer Store: {store_ans_copy}')
                raise NotImplementedError
            goal_node_name = most_common(store_ans)
        
        print('ANS TRIAL:', store_ans)
        print(self.current_state, goal_node_name)

        ########### End Query LLM for Plan #################

        path = self.scene_graph.plan_shortest_paths(self.current_state, goal_node_name)

        # If we are already in the target room, Start local exploration in the room
        # TODO: maybe later implement to query llm for multiple times.
        if path[-1] == self.current_state:
            start_question = "There is a list."
            obj_lst = self.scene_graph.get_obj_in_room(self.current_state)
            Obs_obj_Discript = "["+ ", ".join(obj_lst) + "]"
            end_question = f"Please select one object that is most likely located near a {goal}."
            whole_query = start_question + Obs_obj_Discript + end_question
            chat_completion = self.llm.query_local_explore(whole_query)
            complete_response = chat_completion.choices[0].message.content.lower()
            sample_response = complete_response[complete_response.find('sample answer:'):]
            seperate_ans = re.split('\n|; |, | |sample answer:', sample_response)
            seperate_ans = [i for i in seperate_ans if i != '']
            path = seperate_ans # ans should be separate_ans[0]
        
        # if next ubgoal is entrance, we mark it.
        if len(path) > 1:
            print('path', path)
            if self.scene_graph.is_type(path[1], 'entrance'):
                self.last_subgoal = path[1]
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

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.hfov = 120
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    agent_cfg.sensor_specifications = [rgb_sensor_spec]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


TURN_DEGREE = 10.0

agent = sim.initialize_agent(sim_settings["default_agent"])
agent.agent_config.action_space["turn_left"].actuation.amount = TURN_DEGREE
agent.agent_config.action_space["turn_right"].actuation.amount = TURN_DEGREE

nav = NavigatorTeleport()
goal = 'sink'

cv2.namedWindow("Viz")
img_lang_obs = None

while True:
    obs = sim.get_sensor_observations()
    obs_rgb_cv2 =  cv2.cvtColor(obs['color_sensor'], cv2.COLOR_BGR2RGB) 
    cv2.imshow("Viz", obs_rgb_cv2)
    key = cv2.waitKey(0)
    if key == ord('a'):
        agent.act('turn_left')
    elif key == ord('d'):
        agent.act('turn_right')
    elif key == ord('w'):
        agent.act('move_forward')
    elif key == ord('q'):
        break
    elif key == ord('o'):
        img_lang_obs = nav.observe(Image.fromarray(obs['color_sensor']))
        print('------------  Receive Lang Obs   -------------')
        print(img_lang_obs)
    elif key == ord('u'):
        if img_lang_obs == None:
            print('Current Obs is None')
        else:
            print('------------  Curernt Lang Obs   -------------')
            location = img_lang_obs['location']
            obj_lst = img_lang_obs['object'][1]
            print(f'Location: {location}\nObjecet: {obj_lst}')
            nav.update_scene_graph(None, img_lang_obs)
            print('------------  Update Scene Graph   -------------')
            print(nav.scene_graph.print_scene_graph(pretty=False))
    elif key == ord('i'):
        path = nav.plan_path(goal)
        print('-------------  Plan Path --------------')
        print(path)

cv2.destroyAllWindows()