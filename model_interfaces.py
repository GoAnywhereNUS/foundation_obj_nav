import os
import sys
import yaml
import functools
import torch
import torchvision
from PIL import Image

# GPT
import openai
import logging
# Set up logging
logging.basicConfig(filename='log/llm.log', level=logging.INFO, format='%(asctime)s | %(levelname)s: %(message)s')

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


######## Interface classes ########

class LLMInterface:
    def __init__(self):
        pass

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
Sample Answer: door_1"""

USER_EXAMPLE_2 = """You see the partial layout of the apartment:
{"room": {"kitchen_1": {"connects to": ["door_1", "door_2"]}, "bedroom": {"connects to": ["door_2"]}, "entrance": {"door_1": {"is near": ["towel_1"]}, "door_2": {"is near": [], "connects to": ["kitchen_1","bedroom" ]}}}
Question: Your goal is to find a refrigerator. If any of the rooms in the layout are likely to contain the target object, specify the most probable room name. If all the room are not likely contain the target object, provide the door you would select for exploring a new room where the target object might be found."""

AGENT_EXAMPLE_2 = """Reasoning: There are kitchen and bedroom in the layout. Among all the rooms, kitchen is usually likely to contain refrigerator. Since we haven't explored the kitchen yet, it is possible that the refrigerator is in the kitchen yet. Therefore, I will explore kitchen. 
Sample Answer: kitchen_1"""

##############################

CLS_USER_EXAMPLE_1 = """There is a list ["livingroom", "door", "doorway", "table","chair","livingroom sofa", "floor", "wall"]. Please eliminate redundant strings in the element from the list and classify them into "room," "entrance," and "object" classes."""

CLS_AGENT_EXAMPLE_1 = """Sample Answer:
room: livingroom
entrance: door, doorway
object: table, chair, sofa, floor, wall"""

CLS_USER_EXAMPLE_2 = """There is a list ["bathroom", "bathroom mirror","bathroom sink","toilet", "bathroom bathtub", "lamp"]. Please eliminate redundant strings in the element from the list and classify them into "room," "entrance," and "object" classes."""

CLS_AGENT_EXAMPLE_2 = """Sample Answer:
room: bathroom
entrance: none
object: mirror, sink, toilet, bathtub, lamp"""
#############################

LOCAL_EXP_USER_EXAMPLE_1 = """There is a list ["mirror", "lamp", "picture", "tool","toilet","sofa", "floor", "wall"]. Please select one object that is most likely located near a sink."""

LOCAL_EXP_AGENT_EXAMPLE_1 = """Reasoning: Among the given options, the object most likely located near a sink is a "mirror." Mirrors are commonly found near sinks in bathrooms for personal grooming and hygiene activities.
Sample Answer: mirror"""

LOCAL_EXP_USER_EXAMPLE_2 = """There is a list ["chair", "sofa", "bed", "dresser","ceiling","closet", "window", "wall"]. Please select one object that is most likely located near a table."""

LOCAL_EXP_AGENT_EXAMPLE_2 = """Reasoning: Among the given options, the object most likely located near a table is a "chair." Chairs are commonly placed around tables for seating during various activities such as dining, working, or socializing.
Sample Answer: chair"""

#######################

class GPTInterface(LLMInterface):
    def __init__(
        self,
        key_path="configs/openai_api_key.yaml",
        config_path="configs/gpt_config.yaml",
        log_path ='log/llm_query.log'
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
            {"role": "assistant", "content": AGENT_EXAMPLE_2}
        ]

    def reset(self):
        self.chat = [
            {"role": "system", "content": self.config["setup_message"]},
            {"role": "user", "content": USER_EXAMPLE_1},
            {"role": "assistant", "content": AGENT_EXAMPLE_1},
            {"role": "user", "content": USER_EXAMPLE_2},
            {"role": "assistant", "content": AGENT_EXAMPLE_2}
        ]

    def query(self, string):
        self.reset()
        self.chat.append(
            {"role": "user", "content": string}
        )
        print('QUERY MESSGAE', self.chat)

        response = self.client.chat.completions.create(
            model=self.config["model_type"],
            messages=self.chat,
            seed=self.config["seed"]
        )
        logging.info(f'QUERY MESSAGE: {self.chat}')
        log_reply =  response.choices[0].message.content.replace("\n", ";")
        logging.info(f'REPLY MESSAGE: {log_reply}')
        return response
    
    def query_local_explore(self, string):
        local_exp_query = [
            {"role": "system", "content": self.config["setup_message"]},
            {"role": "user", "content": LOCAL_EXP_USER_EXAMPLE_1},
            {"role": "assistant", "content": LOCAL_EXP_AGENT_EXAMPLE_1},
            {"role": "user", "content": LOCAL_EXP_USER_EXAMPLE_2},
            {"role": "assistant", "content": LOCAL_EXP_AGENT_EXAMPLE_2},
            {"role": "user", "content": string}
        ]
        print('QUERY MESSGAE', local_exp_query)
        response = self.client.chat.completions.create(
            model=self.config["model_type"],
            messages=local_exp_query,
            seed=self.config["seed"]
        )
        logging.info(f'QUERY MESSAGE: {local_exp_query}')
        log_reply =  response.choices[0].message.content.replace("\n", ";")
        logging.info(f'REPLY MESSAGE: {log_reply}')
        return response

    def query_object_class(self, string):
        chat_query_obj = [
            {"role": "system", "content": self.config["setup_message"]},
            {"role": "user", "content": CLS_USER_EXAMPLE_1},
            {"role": "assistant", "content": CLS_AGENT_EXAMPLE_1},
            {"role": "user", "content": CLS_USER_EXAMPLE_2},
            {"role": "assistant", "content": CLS_AGENT_EXAMPLE_2},
            {"role": "user", "content": string}
        ]
        print('QUERY MESSGAE', chat_query_obj)
        response = self.client.chat.completions.create(
            model=self.config["model_type"],
            messages=chat_query_obj,
            seed=self.config["seed"]
        )
        logging.info(f'QUERY MESSAGE: {chat_query_obj}')
        log_reply =  response.choices[0].message.content.replace("\n", ";")
        logging.info(f'REPLY MESSAGE: {log_reply}')
        return response


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
    
    
class VLM_GroundingDino(ObjectPerception):
    def __init__(
        self,
        groundingdino_config_path="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ram_ckpt_path="/home/zhanxin/Desktop/mount/ram_swin_large_14m.pth",
        groundingdino_ckpt_path="/home/zhanxin/Desktop/mount/groundingdino_swint_ogc.pth",
        tag2text_ckpt_path="/home/zhanxin/Desktop/mount/tag2text_swin_14m.pth",
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
        return boxes_filt, selected_labels

    def detect_all_objects(self, image):
        return self._detect_objects(image)
    
    def detect_specific_objects(self, image, object_list):
        assert len(object_list) > 0, "If not detecting specific objects, use detect_all_objects"
        additional_tags = functools.reduce(lambda a, b: a + ", " + b, object_list)
        return self._detect_objects(image, additional_tags)


if __name__ == "__main__":
    image = Image.open("/home/zhanxin/Desktop/SceneGraph/obs.png")

    blip = VLM_BLIP()
    output = blip.query(image, "Where is the photo taken?")
    print(output)

    gdino = VLM_GroundingDino()
    boxes, labels = gdino.detect_all_objects(image)
    print(boxes)
    print(labels)


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
    # sample_response = complete_response[complete_response.find('sample answer:'):]
    # seperate_ans = re.split('\n|; |, | |sample answer:', sample_response)
    # seperate_ans = [i for i in seperate_ans if i != '']
    # print(seperate_ans) # ans should be separate_ans[0]