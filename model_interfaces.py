import os
import sys
import yaml
import functools
import torch
import torchvision
from PIL import Image
import numpy as np
import re
# GPT
import openai
import logging

# BLIP
from lavis.models import load_model_and_preprocess

# GroundingDINO
import groundingdino.datasets.transforms as GDT
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import importlib
tag2text_path = os.path.join(os.getcwd(), 'Grounded-Segment-Anything/Tag2Text')
sys.path.append(tag2text_path)
tag2text = importlib.import_module("Grounded-Segment-Anything.Tag2Text.models.tag2text")
inference_ram = importlib.import_module("Grounded-Segment-Anything.Tag2Text.inference_ram")
sys.path.remove(tag2text_path)

# LLAVA
try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    print('Loading LLAVA')
except:
    print('Not Loading LLAVA.')


######## Interface classes ########
class LLMInterface:
    def __init__(self):
        self.log_path = None

    def reset(self):
        """
        Resets state of the planner, including clearing LLM context
        """
        raise NotImplementedError


class VQAPerception:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = None
        self.image_preprocessors = None
        self.text_preprocessors = None

    def query(self, image, question_prompt):
        """
        Input:
            image: PIL image type
            question_prompt: String, eg: "Where is the photo taken?"
        
        Return:
            string
        """
        raise NotImplementedError
    

class ObjectPerception:
    def __init__(self):
        self.device = torch.device("cuda")

    def detect_all_objects(self, image):
        """
        Input:
            image: PIL image type
        
        Return:
            list of bounding boxes
        """
        raise NotImplementedError
    
    def detect_specific_objects(self, image, object_list):
        """
        Input:
            image: PIL image type
            object_list: List of String describing objects to find, eg: ["door, "exit", ...]
        
        Return:
            list of bounding boxes
        """
        raise NotImplementedError
  
######## Foundation model instantiations ########

USER_EXAMPLE_1 = """You see the partial layout of the apartment:
{"room": {"livingroom_1", "connects to": ["door_1", "door_2"]}, "diningroom_1": {,"connects to": ["door_1"]}}, "entrance": {"door_1": {"is near": ["towel_1"], "connects to": ["livingroom_1", "diningroom_1"]}, "door_2": {"is near": [], "connects to": ["livingroom_1"]}}}
Question: Your goal is to find a sink. If any of the rooms in the layout are likely to contain the target object, specify the most probable room name. If all the room are not likely contain the target object, provide the door you would select for exploring a new room where the target object might be found."""

AGENT_EXAMPLE_1 = """Reasoning: There is only livingroom in the layout. livingroom is not likely to contain sink, so I will not explore the current room. Among all the doors, door1 is near to towel. A towel is usually more likely to near the bathroom or kitchen, so it is likely that if you explore door1 you will find a bathroom or kitchen and thus find a sink.
Answer: door_1"""

USER_EXAMPLE_2 = """You see the partial layout of the apartment:
{"room": {"livingroom_1": {"connects to": ["doorway_1", "door_2"]}, "entrance": {"doorway_1": {"is near": ["table_1"]}, "door_2": {"is near": ["clock_1"], "connects to": ["livingroom_1" ]}}}
Question: Your goal is to find a oven. If any of the rooms in the layout are likely to contain the target object, specify the most probable room name. If all the room are not likely contain the target object, provide the door you would select for exploring a new room where the target object might be found."""

AGENT_EXAMPLE_2 = """Reasoning: There are only livingroom in the layout. Among all the rooms, livingroom is usually unlikely to contain oven, making it less likely for me to find oven in the current room. Instead, I plan to explore other rooms connected to current living room via entrances. Evaluating the entrances, doorway1 stands out as it is close to a table. Tables are commonly found in kitchens, which often contain ovens. Therefore, I have decided to explore through doorway_1.
Answer: doorway_1"""

USER_EXAMPLE_3 = """You see the partial layout of the apartment:
{"room": {"kitchen_1": {"connects to": ["stair_1"]}, "livingroom": {"connects to": ["stair_1"]}, "entrance": {"stair_1": {"is near": []}}}}
Question: Your goal is to find a sink. If any of the rooms in the layout are likely to contain the target object, specify the most probable room name. If all the room are not likely contain the target object, provide the door you would select for exploring a new room where the target object might be found."""

AGENT_EXAMPLE_3 = """Reasoning: There are kitchen and livingroom in the layout. Among all the rooms, kitchen is usually likely to contain sink. Since we haven't explored the kitchen yet, it is possible that the sink is in the kitchen. Therefore, I will explore kitchen. 
Answer: kitchen_1"""

USER_EXAMPLE_4 = """You see the partial layout of the apartment:
{"room": {"livingroom_1": {"connects to": []}}}
Question: Your goal is to find a sink. If any of the rooms in the layout are likely to contain the target object, specify the most probable room name. If all the room are not likely contain the target object, provide the door you would select for exploring a new room where the target object might be found."""

AGENT_EXAMPLE_4 = """Reasoning: There is only livingroom in the layout. livingroom is not likely to contain sink, but there is no entrance/door/doorway in the layout. Therefore, there is nothing to explore, I have to reply.
Answer: None"""


##############################

CLS_USER_EXAMPLE_1 = """There is a list: ["livingroom_0", "hallway_1", "window_13","door_2", "door frame_18", "doorway_3", "table_4","chair_5","livingroom sofa_6", "floor_7", "wall_8", "doorway_9", "stairs_10"]. Please eliminate redundant strings in the element from the list and classify them into "room," "entrance," and "object" classes. Ignore floor, ceiling and wall."""

CLS_AGENT_EXAMPLE_1 = """Answer:
room: livingroom_0
entrance: door_2, doorway_3, hallway_1, doorway_9, stairs_10, door frame_18
object: table_4, chair_5, sofa_6, window_13"""

CLS_USER_EXAMPLE_2 = """There is a list: ["bathroom_0", "bathroom mirror_1","bathroom sink_2","toilet_3", "bathroom bathtub_4", "lamp_5", "ceiling_10"]. Please eliminate redundant strings in the element from the list and classify them into "room," "entrance," and "object" classes. Ignore floor, ceiling and wall."""

CLS_AGENT_EXAMPLE_2 = """Answer:
room: bathroom_0
entrance: none
object: mirror_1, sink_2, toilet_3, bathtub_4, lamp_5"""
#############################

LOCAL_EXP_USER_EXAMPLE_1 = """There is a list: ["mirror_2", "lamp_1", "picture_7", "tool_6","toilet_8","sofa_11", "floor_12", "wall_13"]. Please select one object that is most likely located near a sink."""

LOCAL_EXP_AGENT_EXAMPLE_1 = """Reasoning: Among the given options, the object most likely located near a sink is a "mirror." Mirrors are commonly found near sinks in bathrooms for personal grooming and hygiene activities.
Answer: mirror_2"""

LOCAL_EXP_USER_EXAMPLE_2 = """There is a list: ["chair_4", "sofa_2", "bed_9", "dresser_1","ceiling_6","closet_5", "window_7", "wall_10"]. Please select one object that is most likely located near a table."""

LOCAL_EXP_AGENT_EXAMPLE_2 = """Reasoning: Among the given options, the object most likely located near a table is a "chair." Chairs are commonly placed around tables for seating during various activities such as dining, working, or socializing.
Answer: chair_4"""

#######################

STATE_EST_USER_EXAMPLE_1 = """Description1: On the left, I can see a brown wood headboard, white paper pillow. On the right, I can see a black metal television, gray plastic laundry basket, white wood closet dresser, brown wood drawer. In fron of me, I can see a white wood bed, white wood window, brown metal lamp, brown wood dresser, brown wood dresser nightstand, black silk curtain, white plastic curtain, white metal wall lamp, brown wood drawer. Behind me, I can see a brown wood cabinet. 
Description2: On the left, I can see a white wood door. On the right, I can see a white wood bed, white glass lamp, white glass window, white plastic curtain, brown wood dresser nightstand, white glass window, white wood nightstand, blue fabric curtain, white cotton pillow, white metal ceiling fan, silver metal wall lamp. In front of me, I can see a white glass lamp, brown wood headboard, white cotton pillow, brown wood dresser, white wood bed, white cotton pillow, red metal wall lamp door, brown wood drawer. Behind me, I can see a brown wood bureau, black glass television, brown wood stool, brown wood drawer, brown wood drawer, brown wood drawer.
These are depictions of what I observe from two different vantage points. Please assess the shared objects and spatial relationship in the descriptions to determine whether these two positions are indeed in the same place. Provide a response of True or False, along with supporting reasons. In each direction, focus on only two to three large objects for reasoning.
"""
STATE_EST_AGENT_EXAMPLE_1 = """Reasoning: To simplify the description, given the abundance of objects, we initially focus on common perceptions of object sizes. Our attention is directed toward larger objects, as these are less prone to detection errors.
Description 1: On the left, there is brown wood headboard, white paper pillow. On the right, black metal television, brown wood dresser, and white wood closet dresser are of relatively large size. In front of me, there is a white wood bed and a brown wood dresser. Behind me, I can see a brown wood cabinet and black silk curtain.
Description 2: On the left, there is white wood door. On the right, the white wood bed, brown wood dresser, nightstand, and ceiling fan are of relatively large size. In front of me, there is a white wood bed, black glass television a brown wood dresser. Behind me, I can see a brown wood cabinet.
Shared Large Objects: the two descriptions exhibit significant commonalities, prominently featuring large and easily observable items such as a brown wood dresser, a brown wood drawer, a white wood bed, a metal wall lamp, a television, and a brown wood headboard. Spatial Relationship: the spatial relationships within both descriptions remain consistent, with the dresser and wall lamp positioned near the bed in each scenario. Despite minor variations in the color or material of smaller objects like stools or curtains, these discrepancies appear more likely to stem from observational nuances rather than indicating distinct rooms. 
Answer: True"""

STATE_EST_USER_EXAMPLE_2 = """Description1: On the left, there is a silver metal faucet, white glass mirror, black metal wall lamp, white stainless steel sink, white formica countertop, blue plastic accessory, white wood white door, silver metal faucet, white metal wall lamp, brown wood cabinet, black wood white door doorway, brown wood cabinet, white wood mirror wall. On the right, there is a brown tile floor, white drywall ceiling. In front of me, there is a silver metal faucet, silver glass mirror, white soap soap, white porcelain sink, brown wood cabinet, white porcelain sink, silver metal wall lamp, silver metal wall lamp. At the rear, there is an orange glass lamp, brown wood floor, white porcelain tub, white cotton bed.
Description2:  On the left, there is a white glass mirror, silver metal faucet, white porcelain sink, white wood bathroom sink, white wood bed, brown metal wall lamp. On the right, there are no specified items. In front of me, there is a silver metal faucet, white porcelain sink, white plastic toiletry, silver glass mirror, blue plastic soap toiletry, white white bathroom sink countertop, blue metal wall lamp, black wood bathroom cabinet, black metal wall lamp. Behind me, there is a white porcelain tub, black cloth curtain, white cotton bed, brown wood bed, black cloth curtain, white porcelain bath.
These are depictions of what I observe from two different vantage points. Please assess the shared objects and spatial relationship in the descriptions to determine whether these two positions are indeed in the same place. Provide a response of True or False, along with supporting reasons. In each direction, focus on only two to three large objects for reasoning.
"""

STATE_ESTP_AGENT_EXAMPLE_2 = """Reasoning: To simplify the description, given the abundance of objects, we focus initially on common perceptions of larger objects, as they are less prone to detection errors.
Description 1: On the left, there is brown wood cabinet, silver metal faucet, white wood bathroom sink, and white wood bed. On the right, there is nothing. In front of me, there is a silver metal faucet, silver glass mirror,white porcelain sink, brown wood cabinet, black mental wall lamp.  Behind me, there is an orange glass lamp, white porcelain tub, white cotton bed. 
Description 2: On the left, there is white porcelain sink and white wood bed. On the right, there are nothing. In front of me, there is a silver mental faucet and white plastic toiletry. Behind me, I can see white porcelain tub, black cloth curtain, white cotton bed.
Shared Large Objects: Both descriptions mention silver metal faucets, white beds, and sink, contributing to a consistent thematic presence. Spatial Relationship: The placement of key items like silver metal faucets and brown wood cabinets described in similar configurations create a sense of coherence. Potential for Small Errors: Description 2 lacks specified items, such as accessory, door and countertop. Besides, there are small errors in the material of bed. Errors in material or color, as well as the potential omission of small-size objects, can indeed be attributed to variations in observation. Despite this, the two descriptions are depicting the same room, with the potential for small errors accounting for minor variations.
Answer: True"""

STATE_EST_USER_EXAMPLE_3 = """Description1: On the left, there is a white wood dresser, white glass lamp, and a white glass window/door. On the right, there is a white wood chair and a silver metal wall lamp. In front of me, there is an incomplete description. Behind me, there is a purple wood bed, yellow wood ceiling fan, white wood dresser, and another white glass window.
Description2: On the left, there is a white glass mirror, silver metal faucet, white porcelain sink, white wood bathroom sink, white wood bed, brown metal wall lamp. On the right, there are no specified items. In front of me, there is a silver metal faucet, white porcelain sink, white plastic toiletry, silver glass mirror, blue plastic soap toiletry, white white bathroom sink countertop, blue metal wall lamp, black wood bathroom cabinet, black metal wall lamp. Behind me, there is a white porcelain tub, black cloth curtain, white cotton bed, brown wood bed, black cloth curtain, white porcelain bath.
These are depictions of what I observe from two different vantage points. Please assess the shared objects and spatial relationship in the descriptions to determine whether these two positions are indeed in the same place. Provide a response of True or False, along with supporting reasons. In each direction, focus on only two to three large objects for reasoning.
"""

STATE_ESTP_AGENT_EXAMPLE_3 = """Reasoning: To simplify the description, given the abundance of objects, we focus initially on common perceptions of larger objects, as they are less prone to detection errors.
Description1: On the left, there is a white wood dresser, and a white glass window/door. On the right, there is a white wood chair and a silver metal wall lamp. In front of me, there is nothing. Behind me, there is a purple wood bed, yellow wood ceiling fan, white wood dresser.
Description2: On the left, there is a silver metal faucet, white porcelain sink, white wood sink, white wood bed. On the right, there is nothing. In front of me, there is a silver metal faucet, white porcelain sink, white plastic toiletry. Behind me, there is a white porcelain tub, black cloth curtain, white cotton bed, white porcelain bath.
Shared Large Objects: Both descriptions feature common large objects, including a wood bed and a silver metal wall lamp, though the color of bed is different. Spatial Relationships: Despite variations in surrounding details, the consistent mention of a white wood bed suggests a shared spatial context, possibly from different viewpoints within the same room. Minor differences, such as color variations in the bed, may be attributed to observational nuances rather than indicating distinct rooms, reinforcing the likelihood of the same room.
Answer: True"""
#######################


class GPTInterface(LLMInterface):
    def __init__(
        self,
        key_path="configs/openai_api_key.yaml",
        config_path="configs/gpt_config.yaml"
        ):

        super().__init__()

        with open(key_path, 'r') as f:
            key_dict = yaml.safe_load(f)
            self.openai_api_key = key_dict['api_key']
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.client = openai
        self.client.api_key = self.openai_api_key
        self.chat = [
            {"role": "system", "content": self.config["setup_message"]},
            {"role": "user", "content": USER_EXAMPLE_1},
            {"role": "assistant", "content": AGENT_EXAMPLE_1},
            {"role": "user", "content": USER_EXAMPLE_2},
            {"role": "assistant", "content": AGENT_EXAMPLE_2},
            {"role": "user", "content": USER_EXAMPLE_3},
            {"role": "assistant", "content": AGENT_EXAMPLE_3},
            {"role": "user", "content": USER_EXAMPLE_4},
            {"role": "assistant", "content": AGENT_EXAMPLE_4}
        ]

        logs_folder = 'logs'

        all_folders = [folder for folder in os.listdir(logs_folder) if os.path.isdir(os.path.join(logs_folder, folder))]

        # Filter folders that start with "trial_"
        trial_folders = [folder for folder in all_folders if folder.startswith("trial_")]

        # Extract the numbers and find the maximum
        numbers = [int(folder.split("_")[1]) for folder in trial_folders]
        max_number = max(numbers, default=0)
        trial_folder = os.path.join(logs_folder, 'trial_' + str(max_number))
        log_path = os.path.join(trial_folder, 'llm_query.log')
        
        self.log_path = log_path

    def reset(self, log_path=None):
        if log_path is None:
            logs_folder = 'logs'
            all_folders = [folder for folder in os.listdir(logs_folder) if os.path.isdir(os.path.join(logs_folder, folder))]

            # Filter folders that start with "trial_"
            trial_folders = [folder for folder in all_folders if folder.startswith("trial_")]

            # Extract the numbers and find the maximum
            numbers = [int(folder.split("_")[1]) for folder in trial_folders]
            max_number = max(numbers, default=0)
            log_path = os.path.join(logs_folder, 'trial_' + str(max_number))
            
        log_path = os.path.join(log_path, 'llm_query.log')
        self.log_path = log_path

    def query(self, string):
        self.chat = [
            {"role": "system", "content": self.config["setup_message"]},
            {"role": "user", "content": USER_EXAMPLE_1},
            {"role": "assistant", "content": AGENT_EXAMPLE_1},
            {"role": "user", "content": USER_EXAMPLE_2},
            {"role": "assistant", "content": AGENT_EXAMPLE_2},
            {"role": "user", "content": USER_EXAMPLE_3},
            {"role": "assistant", "content": AGENT_EXAMPLE_3},
            {"role": "user", "content": USER_EXAMPLE_4},
            {"role": "assistant", "content": AGENT_EXAMPLE_4}
        ]
        self.chat.append(
            {"role": "user", "content": string}
        )
        # print('PLAN MESSGAE', string)

        response = self.client.chat.completions.create(
            model=self.config["model_type"],
            messages=self.chat,
            seed=self.config["seed"]
        )
        with open(self.log_path, 'a') as file:
            file.write(f'[PLAN QUERY MESSAGE]: {self.chat}\n')
            log_reply =  response.choices[0].message.content.replace("\n", ";")
            file.write(f'[PLAN REPLY MESSAGE]: {log_reply}\n')
        
        complete_response = response.choices[0].message.content.lower()
        sample_response = complete_response[complete_response.find('answer:'):]
        seperate_ans = re.split('\n|; |, | |answer:', sample_response)
        seperate_ans = [i.replace('.','') for i in seperate_ans if i != ''] # to make sink. to sink
        return seperate_ans
    
    def query_local_explore(self, string):
        local_exp_query = [
            {"role": "system", "content": self.config["setup_message"]},
            {"role": "user", "content": LOCAL_EXP_USER_EXAMPLE_1},
            {"role": "assistant", "content": LOCAL_EXP_AGENT_EXAMPLE_1},
            {"role": "user", "content": LOCAL_EXP_USER_EXAMPLE_2},
            {"role": "assistant", "content": LOCAL_EXP_AGENT_EXAMPLE_2},
            {"role": "user", "content": string}
        ]
        # print('LOCAL MESSGAE', string)
        response = self.client.chat.completions.create(
            model=self.config["model_type"],
            messages=local_exp_query,
            seed=self.config["seed"]
        )
    
        with open(self.log_path, 'a') as file:
            file.write(f'[LOCAL QUERY MESSAGE]: {local_exp_query}\n')
            log_reply =  response.choices[0].message.content.replace("\n", ";")
            file.write(f'[LOCAL REPLY MESSAGE]: {log_reply}\n')
        
        complete_response = response.choices[0].message.content.lower()
        sample_response = complete_response[complete_response.find('answer:'):]
        seperate_ans = re.split('\n|; |, | |answer:', sample_response)
        seperate_ans = [i.replace('.','') for i in seperate_ans if i != '']

        return seperate_ans

    def query_object_class(self, string):
        chat_query_obj = [
            {"role": "system", "content": self.config["setup_message"]},
            {"role": "user", "content": CLS_USER_EXAMPLE_1},
            {"role": "assistant", "content": CLS_AGENT_EXAMPLE_1},
            {"role": "user", "content": CLS_USER_EXAMPLE_2},
            {"role": "assistant", "content": CLS_AGENT_EXAMPLE_2},
            {"role": "user", "content": string}
        ]
        # print('CLASSIFY MESSGAE', string)
        response = self.client.chat.completions.create(
            model=self.config["model_type"],
            messages=chat_query_obj,
            seed=self.config["seed"]
        )
        with open(self.log_path, 'a') as file:
            file.write(f'[CLASSIFY QUERY MESSAGE]: {chat_query_obj}\n')
            log_reply =  response.choices[0].message.content.replace("\n", ";")
            file.write(f'[CLASSIFY REPLY MESSAGE]: {log_reply}\n')
        
        complete_response = response.choices[0].message.content.lower()
        complete_response = complete_response.replace(" ", "")
        seperate_ans = re.split('\n|,|:', complete_response)
        seperate_ans = [i.replace('.','') for i in seperate_ans if i != ''] 
        
        return seperate_ans

    def query_state_estimation(self, string):
        chat_query = [
            {"role": "system", "content": "You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance on where to explore based on a series of observations. Observations will be given as descriptions of objects seen from four cameras in four directions. Your job is to estimate the robot's state. You will be given two descriptions, and you need to decide whether these two descriptions describe the same room. For example, if we have visited the room before and got one description, when we visit a similar room and get another description, it is your job to determine whether the two descriptions represent the same room. You should understand that descriptions may contain errors and noise due to sensor noise and partial observability. Always provide reasoning along with a deterministic answer. If there are no suitable answers, leave the space after 'Answer: None.' Always include: Reasoning: <your reasoning> Answer: <your answer>."},
            {"role": "user", "content": STATE_EST_USER_EXAMPLE_1},
            {"role": "assistant", "content": STATE_EST_USER_EXAMPLE_1},
            {"role": "user", "content": STATE_EST_USER_EXAMPLE_2},
            {"role": "assistant", "content": STATE_ESTP_AGENT_EXAMPLE_2},
            {"role": "user", "content": STATE_EST_USER_EXAMPLE_3},
            {"role": "assistant", "content": STATE_ESTP_AGENT_EXAMPLE_3},
            {"role": "user", "content": string}
        ]
        # print('CLASSIFY MESSGAE', string)
        response = self.client.chat.completions.create(
            model=self.config["model_type"],
            messages=chat_query,
            seed=self.config["seed"]
        )
        with open(self.log_path, 'a') as file:
            file.write(f'[STATE EST QUERY MESSAGE]: {chat_query}\n')
            log_reply =  response.choices[0].message.content.replace("\n", ";")
            file.write(f'[STATE EST REPLY MESSAGE]: {log_reply}\n')
       
        complete_response = response.choices[0].message.content.lower()
        sample_response = complete_response[complete_response.find('answer:'):]
        seperate_ans = re.split('\n|; |, | |answer:', sample_response)
        seperate_ans = [i.replace('.','') for i in seperate_ans if i != '']
        print(complete_response)
        if len(seperate_ans) > 0:
            cleaned_ans = seperate_ans[0]
        else:
            cleaned_ans = 'none'
        return cleaned_ans

class VLM_BLIP(VQAPerception):
    def __init__(self):

        super().__init__()

        self.model, self.image_preprocessors, self.text_preprocessors = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=self.device
        )

    def query(self, image, question_prompt):
        image = image.convert("RGB")
        samples = {
            "image": self.image_preprocessors["eval"](image).unsqueeze(0).to(self.device),
            "text_input": self.text_preprocessors["eval"](question_prompt)
        }

        ans = self.model.predict_answers(samples=samples, inference_method="generate")
        return ans[0]
    
class VLM_LLAVA(VQAPerception):
    def __init__(self):
        super().__init__()
        # LLAVA
        self.model_path = 'liuhaotian/llava-v1.5-7b' # unsure how to change the model path
        self.model_name = 'llava-v1.5-7b'
        self.conv_mode = "llava_v1"
        self.load_4bit = True
        self.load_8bit = False
        self.model_base = None
        self.tokenizer, self.model,  self.image_processor,  self.context_len = load_pretrained_model(self.model_path, self.model_base, self.model_name, self.load_8bit, self.load_4bit, device='cuda') # set decive = self.device cause issues
        self.conv = conv_templates[self.conv_mode].copy()
        self.roles = self.conv.roles

    def query(self, image, question_prompt):
        from transformers import TextStreamer
        self.reset()
        image = image.convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question_prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + question_prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature= 0.2,
                max_new_tokens= 512,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip('</s>')
        self.conv.messages[-1][-1] = outputs
        return outputs

    def reset(self):
        self.conv = conv_templates[self.conv_mode].copy()

class VLM_GroundingDino(ObjectPerception):
    def __init__(
        self,
        groundingdino_config_path="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ram_ckpt_path="checkpoints/ram_swin_large_14m.pth",
        groundingdino_ckpt_path="checkpoints/groundingdino_swint_ogc.pth",
    ):

        super().__init__()

        args = SLConfig.fromfile(groundingdino_config_path)
        args.device = self.device
        gdino_ckpt = torch.load(groundingdino_ckpt_path, map_location="cuda")
        self.box_threshold = 0.25
        self.text_threshold = 0.2
        self.iou_threshold = 0.5

        self.gdino_model = build_model(args)
        self.gdino_model.load_state_dict(clean_state_dict(gdino_ckpt['model']), strict=False)
        self.gdino_model.eval()
        self.gdino_model.to(self.device)

        self.ram_model = tag2text.ram(pretrained=ram_ckpt_path, image_size=384, vit='swin_l')
        self.ram_model.eval()
        self.ram_model.to(self.device)

        self.preprocessor_gdino = GDT.Compose(
            [
                GDT.RandomResize([800], max_size=1333),
                GDT.ToTensor(),
                GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.preprocessor_ram = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (384, 384), torchvision.transforms.InterpolationMode.BICUBIC
            ),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess_image(self, image, model):
        """
        Image preprocessor.

        Input:
            image: PIL image
        
        Return:
            image_rgb: PIL Image (RGB), the original image in RGB format
            processed: PIL Image (RGB)
        """
        image_rgb = image.convert("RGB")
        if model == "gdino":
            processed, _ = self.preprocessor_gdino(image_rgb, None)  # 3, h, w
        elif model == "ram":
            processed = self.preprocessor_ram(image_rgb)  # 3, h, w
        else:
            raise NotImplementedError
        return processed
    
    def _run_gdino(self, image, caption):
        # Format tags
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        # Run inference on GDino model
        with torch.no_grad():
            outputs = self.gdino_model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenized = self.gdino_model.tokenizer(caption)

        # build pred
        pred_phrases = []
        scores = []
        selected_labels = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > self.text_threshold, tokenized, self.gdino_model.tokenizer
            )
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            selected_labels.append(pred_phrase)
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases, selected_labels
    
    def _detect_objects(self, image, additional_tags=""):
        gdino_image = self._preprocess_image(image, "gdino").to(self.device)
        ram_image = self._preprocess_image(image, "ram").unsqueeze(0).to(self.device)

        # RAM
        tags, _ = inference_ram.inference(ram_image, self.ram_model)
        tags = tags.replace(' |', ',')
        tags += additional_tags

        # GroundingDINO
        boxes_filt, scores, pred_phrases, selected_labels = self._run_gdino(
            gdino_image, tags
        )
        
        # Scale normalized bounding boxes to original image size
        w, h = image.size
        boxes_filt = boxes_filt.cpu()
        for i in range(len(boxes_filt)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([w, h, w, h])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        # use NMS to handle overlapped boxes
        nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        selected_labels = [selected_labels[idx] for idx in nms_idx]
        cropeed_obj_lst = []
        for i in range(len(boxes_filt)):
            cropped_img = image.crop(np.array(boxes_filt[i]))
            cropeed_obj_lst.append(cropped_img)

        return boxes_filt, selected_labels, cropeed_obj_lst

    def detect_all_objects(self, image):
        return self._detect_objects(image)
    
    def detect_specific_objects(self, image, object_list):
        assert len(object_list) > 0, "If not detecting specific objects, use detect_all_objects"
        additional_tags = functools.reduce(lambda a, b: a + ", " + b, object_list)
        return self._detect_objects(image, additional_tags)


if __name__ == "__main__":
    import pdb
    # image = Image.open("/home/zhanxin/foundation_obj_nav/room_test3.png")

    # gdino = VLM_GroundingDino()
    # boxes, labels = gdino.detect_all_objects(image)
    # print(boxes)
    # print(labels)

    # blip = VLM_BLIP()
    # output = blip.query(image, "Which room is the photo?")
    # print(output)

    # folder_path = "/home/zhanxin/foundation_obj_nav/data/rls_tour_fisheye"

    # # List all files in the folder
    # files = os.listdir(folder_path) # file name list
    # import matplotlib.pyplot as plt
    # # Read and process each image
    # for image_file in files:
    #     image_path = os.path.join(folder_path, image_file)
    #     image = Image.open(image_path)
    #     output = blip.query(image, "Which room is the photo?")
    #     boxes, labels, _ = gdino.detect_all_objects(image)
    #     plt.imshow(image)
    #     ax = plt.gca()
        
    #     for i in range(len(labels)):
    #         label = labels[i]
    #         min_x, min_y, max_x, max_y = boxes[i]
    #         ax.add_patch(plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    #         ax.text(min_x, min_y, label)
    #     plt.savefig(os.path.join(folder_path, image_file[:-4] + "_" + output + ".png") )
    #     plt.clf()


    llm_config_path="configs/gpt_config.yaml"
    llm = GPTInterface(config_path=llm_config_path)

    role = "You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance on where to explore based on a series of observations. Observations will be given as descriptions of objects seen from four cameras in four directions. Your job is to estimate the robot's state. You will be given two descriptions, and you need to decide whether these two descriptions describe the same room. For example, if we have visited the room before and got one description, when we visit a similar room and get another description, it is your job to determine whether the two descriptions represent the same room. You should understand that descriptions may contain errors and noise due to sensor noise and partial observability. Always provide reasoning along with a deterministic answer. If there are no suitable answers, leave the space after 'Answer: None.' Always include: Reasoning: <your reasoning> Answer: <your answer>."
    # import re
    discript1 = "Description 1: On the left, there is wallhangingdecoration. On the right, there is nothing. In front of me, there is a white wood stool and a brown tile floor. Behind me, there is a purple cotton bed, a white glass window, a white Lego dresser, a white metal wall lamp, a white metal lamp, a white glass window, a blue drywall wall, and another white glass window.\n"
    # # discript2 = "Description 2: On the left, there is purple wood bed, black metal walllamp, purple and white cotton pillow, white glass window, white metal ceilingfan. On the right, there is white glass window, brown wood cart. On the forward, there is . On the rear, there is purple cotton bed, white glass window, white glass window, black metal walllamp\n"
    # discript2 = "Description 2: On the left, there is nothing. On the right, there is nothing. In front of me, there is a wood stool and a tiled floor. Behind me, there is a violet bed, a white glass window, a cream dresser, a white wall lamp, a white metal lamp, a white glass window, a blue wall, and another white glass window.\n"
    # discript2 = "Description 2: On the left, there is nothing. On the right, there is nothing. In front of me, there is a tiled floor and a wood stool. Behind me, there is a violet bed, a white glass window, a white wall lamp, a cream dresser, a white metal lamp, a blue wall, a white glass window, and another white glass window.\n"
    discript2 = "Description 2: On the left, there is pillow, computer. On the right, there is a book. In front of me, there is a wood stool. Behind me, there is a purple violet bed, a white glass window, a white wall lamp, a stool, a cream dresser, a white metal lamp, a blue wall, and a white glass window.\n"
    
    # discript1 = "Description 1: There is wallhangingdecoration, a white wood stool, a brown tile floor, a purple cotton bed, a white glass window, a white Lego dresser, a white metal wall lamp, a white metal lamp, a white glass window, a blue drywall wall, and another white glass window.\n"
    # discript2 = "Description 2: There is pillow, a book, a wood stool, a purple violet bed, a white glass window, a white wall lamp, a stool, a cream dresser, a white metal lamp, a blue wall, and a white glass window.\n"
    
    # discript1 = "Description 1: On the left, there is painting, pillow, chair. On the right, there is wallhangingdecoration, photo, bathtub. In front of me, there is clothes, pillow, bedtable, blinds. Behind me, there is curtainrail, shoes."
    # discript2 = "Description 2: On the left, there is picture, brush. On the right, there is pillow, wallhangingdecoration.In front of me, there is painting, bathtub, white cotton pillow, chair. Behind me, there is basketofsomething, shoes, clothes"
    # discript1 = "Description 1: On the left, there is glass painting, floorlamp, wallhangingdecoration, book, pillow, pillow, pillow, chair, blinds, windowframe. On the right, there is wallhangingdecoration, paper photo,photo, plastic photo, pillow,  pillow, pillow, pillow, pillow, pillow, bedtable, bedsidelamp, book, blinds, windowframe, doorframe, white towel, bathtub. In front of me, there is ceilingvent, pillow, black and blanket, wallhangingdecoration, footstool, bed,bedsidelamp, bedtable, windowframe,nblinds, windowframe, blinds, windowframe. Behind me, there is curtainrail"
    # discript2 = "Description 2: On the left, there is picture, brush, brush. On the right, there are pillow, wallhangingdecoration, wallhangingdecoration, wallhangingdecoration, wallhangingdecoration, bed, ceilingvent, towel. In front of me, there is painting, chandelier, pillow, pillow, curtain, clothes,  box, bag, shelf, clotheshanger, mirror, foodstand, diningtable. Behind me, there is cotton basketofsomething, shoes, clothes"
 
    # discript1 = "Description 1: You see glass painting, floorlamp, wallhangingdecoration, book, pillow, pillow, pillow, chair, blinds, windowframe,  wallhangingdecoration, photo, photo, photo, pillow,  pillow, pillow, pillow, pillow, pillow, bedtable, bedsidelamp, book, blinds, windowframe, doorframe, towel, bathtub, ceilingvent, pillow, black and blanket, wallhangingdecoration, footstool, bed,bedsidelamp, bedtable, windowframe,nblinds, windowframe, blinds, windowframe and curtainrail"
    # discript2 = "Description 2: You see picture, brush, brush, pillow, wallhangingdecoration, wallhangingdecoration, wallhangingdecoration, wallhangingdecoration, bed, ceilingvent, towel, painting, chandelier, pillow, pillow, curtain, clothes,  box, bag, shelf, clotheshanger, mirror, foodstand, diningtable, cotton basketofsomething, shoes, clothes"
 
    question = "These are depictions from two different vantage points. Please assess the shared objects and spatial relationship in the descriptions to determine whether these two descriptions represent the same room. Provide a response of True or False, along with supporting reasons. If you cannot decide, reply None in answer, but please aim for a conclusive response. To simplify the description, focus on larger objects."
    whole_query = role + discript1 + discript2 + question
    store_ans = []
    for i in range(20):
        chat_completion = llm.query_state_estimation(whole_query)
        store_ans.append(chat_completion)
        print(chat_completion)
    # # pdb.set_trace()

    print(store_ans)

    #### ---------------------   LOCAL EXPLOARATION TEST  -------------------------
    # Notice: add open_ai key config before test

    # llm_config_path="configs/gpt_config.yaml"
    # llm = GPTInterface(config_path=llm_config_path)

    # goal = "sofa"
    # start_question = "There is a list."
    # Obs_obj_Discript = "["+ ", ".join(obs['object']) + "]"
    # end_question = f"Please select one object that is most likely located near a {goal}."
    # whole_query = start_question + Obs_obj_Discript + end_question

    # chat_completion = llm.query_local_explore(whole_query)
    # complete_response = chat_completion.choices[0].message.content.lower()
    # sample_response = complete_response[complete_response.find('answer:'):]
    # seperate_ans = re.split('\n|; |, | |answer:', sample_response)
    # seperate_ans = [i for i in seperate_ans if i != '']
    # print(seperate_ans) # ans should be separate_ans[0]
    
    # llava = VLM_LLAVA()
    # import time
    # for i in range(10):
    #     start = time.time()
    #     output = llava.query(image, "Where is the photo taken? Please reply with one word.")
    #     print(output)
    #     end = time.time()
    #      print('Cost Time:', end-start)
    #     start = time.time()
    #     output = llava.query(image, "List objects in the photo with as many details as possible")
    #     print(output)
    #     end = time.time()
    #     print('Cost Time:', end-start)
