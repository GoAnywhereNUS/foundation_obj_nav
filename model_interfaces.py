import os
import sys
import yaml
import torch
import torchvision
from PIL import Image
import numpy as np
import re
from functools import reduce
from typing import Any, Callable, Optional, Union

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

######## Abstract driver classes ########

class BaseLLMDriver:
    def __init__(self):
        self.device = torch.device('cuda')

    def reset(self):
        raise NotImplementedError("Subclasses to implement this method")
    
    def send_query(self, prompt: str, required_samples: int):
        raise NotImplementedError("Subclasses to implement this method")
    
class BaseVQADriver:
    def __init__(self):
        self.device = torch.device('cuda')

    def reset(self):
        raise NotImplementedError("Subclasses to implement this method")
    
    def send_query(
        self,
        image_input: Union[Image.Image, list],
        prompt_input: Union[str, list],
        prompts_per_image: int,
    ):
        raise NotImplementedError("Subclasses to implement this method")
    
class BaseObjectDriver:
    def __init__(self):
        self.device = torch.device('cuda')

    def reset(self):
        raise NotImplementedError("Subclasses to implement this method")
    
    def send_query(
        self,
        image: Image.Image,
        additional_tags: list[str],
    ):
        raise NotImplementedError("Subclasses to implement this method")
    
######## Driver classes that directly run models ########

class ModelLLMDriver_GPT(BaseLLMDriver):
    def __init__(
        self,
        key_path="configs/openai_api_key.yaml",
        config_path="configs/gpt_config.yaml",
    ):
        super().__init__()
        with open(key_path, 'r') as f:
            key_dict = yaml.safe_load(f)
            api_key = key_dict['api_key']
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.client = openai
        self.client.api_key = api_key
        print("Set up LLM driver!")

    def send_query(self, prompt, required_samples):
        response = self.client.chat.completions.create(
            model=self.config["model_type"],
            messages=prompt,
            n=required_samples,
            seed=self.config["seed"],
            temperature=self.config["temperature"]
        )
        return response
        

class ModelVQADriver_BLIP(BaseVQADriver):
    def __init__(self, max_batch_size: int = 16):
        super().__init__()
        self.model, self.image_preprocessors, self.text_preprocessors = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=self.device
        )
        self.max_batch_size = max_batch_size
        print("Set up VQA model!")

    def send_query(
        self,
        image_input: Union[Image.Image, list],
        prompt_input: Union[str, list],
        prompts_per_image: int,
    ):
        preprocessed_ims = torch.stack([
            self.image_preprocessors["eval"](im.convert("RGB")) for im in image_input
        ]).to(self.device)
        if prompts_per_image > 1:
            preprocessed_ims = torch.repeat_interleave(
                preprocessed_ims, prompts_per_image, dim=0)
        preprocessed_qns = [self.text_preprocessors["eval"](p) for p in prompt_input]
        
        batched_iters = int(np.ceil(preprocessed_ims.shape[0] / self.max_batch_size))
            
        answers = []
        for iter in range(batched_iters):
            lo = iter * self.max_batch_size
            hi = min(lo + self.max_batch_size, preprocessed_ims.shape[0])
            samples = {
                "image": preprocessed_ims[lo:hi],
                "text_input": preprocessed_qns[lo:hi],
            }
            answers += self.model.predict_answers(
                samples=samples, inference_method="generate")
            
        return answers

class ModelObjectDriver_GroundingDINO(BaseObjectDriver):
    def __init__(
        self,
        gdino_config_path="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ram_ckpt_path="checkpoints/ram_swin_large_14m.pth",
        gdino_ckpt_path="checkpoints/groundingdino_swint_ogc.pth",
    ):

        super().__init__()

        args = SLConfig.fromfile(gdino_config_path)
        args.device = self.device
        gdino_ckpt = torch.load(gdino_ckpt_path, map_location="cuda")
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

        # Hyperparameters for filtering
        self.small_object_filter_pix = 80
        self.glass_objects_iou_pix = 0.7

        print("Set up GDino model!")

    def send_query(
        self,
        image: Image.Image,
        additional_tags: str,
    ):
        tags = self._caption_objects_in_image(image, additional_tags)
        detections = self._detect_objects_in_tags(image, tags)
        boxes_filt, scores, pred_phrases, selected_labels = detections
        
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
        cropped_ims = []
        for i in range(len(boxes_filt)):
            cropped_img = image.crop(np.array(boxes_filt[i]))
            cropped_ims.append(cropped_img)

        return boxes_filt, selected_labels, cropped_ims
        

    def _caption_objects_in_image(self, image, additional_tags):
        ram_image = self._preprocess_image(image, "ram").unsqueeze(0).to(self.device)
        tags, _ = inference_ram.inference(ram_image, self.ram_model)
        tags = tags.replace(' |', ',')
        tags += additional_tags
        return tags

    def _detect_objects_in_tags(self, image, caption):
        # Preprocess image
        image = self._preprocess_image(image, "gdino").to(self.device)

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

    def _preprocess_image(self, image, model):
        image_rgb = image.convert("RGB")
        if model == "gdino":
            processed, _ = self.preprocessor_gdino(image_rgb, None)  # 3, h, w
        elif model == "ram":
            processed = self.preprocessor_ram(image_rgb)  # 3, h, w
        else:
            raise ValueError(f"Invalid model {model} with no image preprocessor!")
        return processed

    def _filter(self, object_detections):
        od = list(zip(*object_detections))

        # Remove small objects
        filter_small = [
            ((min_x, min_y, max_x, max_y), l, c)
            for (min_x, min_y, max_x, max_y), l, c in od
            if (abs(max_x - min_x) < self.small_object_filter_pix or
                abs(max_y - min_y) < self.small_object_filter_pix)
        ]

        # Remove objects obscured by glass
        glass_objects = torch.Tensor([
            list(bb) + [(bb[2] - bb[0]) * 0.5, (bb[3] - bb[1]) * 0.5]
            for bb, l, _ in filter_small if 'glass' in l
        ])
        bboxes = torch.Tensor([
            list(bb) + [(bb[2] - bb[0]) * 0.5, (bb[3] - bb[1]) * 0.5]
            for bb, _l, _ in filter_small if 'glass' not in l
        ])
        iou_with_glass = torchvision.ops.box_iou(
            bboxes[:, :-2], glass_objects[:, :-2])
        iou_thresholded = torch.all(
            iou_with_glass < self.glass_objects_iou_pix, dim=1)
        filter_glass = [
            obj for obj, low_overlap in zip(filter_small, iou_thresholded) 
            if low_overlap
        ]
        return list(zip(*filter_glass))

######## Model interface classes ########

class LLMInterface:
    def __init__(
        self,
        driver,
    ):
        self.driver = driver

    def reset(self):
        """
        Resets state of the planner, including clearing LLM context
        """
        raise NotImplementedError("Subclasses to implement this method")
    
    # TODO: Implement a batched version of query
    def query(
        self,
        prompt: str,
        validate_fn: Callable[[list[Any]], list[Any]],
        required_samples: int = 1,
        max_tries: int = 3,
    ):
        answers = []
        valid = False
        remaining_samples_needed = required_samples
        for _ in range(max_tries):
            response = self.driver.send_query(prompt, remaining_samples_needed)

            for choice in response.choices:
                validated_resp = validate_fn(choice.message.content)
                if validated_resp is not None:
                    answers.append(validated_resp)

            remaining_samples_needed = required_samples - len(answers)
            valid = remaining_samples_needed <= 0
            if valid:
                break

        return valid, answers


class VQAPerception:
    def __init__(self, driver: type[BaseVQADriver]):
        self.driver = driver

    def query(
            self, 
            image_input: Union[Image.Image, list],
            prompt_input: Union[str, list],
            validate_fn: Callable[[str], Optional[str]],
            prompts_per_image: int = 1,
    ):
        """
        Input:
            image: PIL image type
            question_prompt: String, eg: "Where is the photo taken?"
        
        Return:
            string
        """
        assert prompts_per_image >= 1, "Invalid number of prompts per image"
        assert isinstance(image_input, list) == isinstance(prompt_input, list), \
            "Image and prompt input types do not match"
        if not isinstance(image_input, list):
            image_input, prompt_input = [image_input], [prompt_input]
        assert len(image_input) * prompts_per_image == len(prompt_input), \
            "No. of input images does not match no. of prompts"

        resp = self.driver.send_query(
            image_input, prompt_input, prompts_per_image
        )
        return validate_fn(resp)
    

class ObjectPerception:
    def __init__(self, driver: type[BaseObjectDriver]):
        self.driver = driver

    def detect_all_objects(self, image, filter=False):
        """
        Input:
            image: PIL image type
        Return:
            list of bounding boxes
        """
        object_detections = self._detect_objects(image, "")
        if filter:
            return self._filter(object_detections)
        return object_detections
    
    def detect_specific_objects(self, image, object_list, filter=False):
        """
        Input:
            image: PIL image type
            object_list: List of String describing objects to find, eg: ["door, "exit", ...]
        Return:
            list of bounding boxes
        """
        assert len(object_list) > 0, "If not detecting specific objects, use detect_all_objects"
        additional_tags = reduce(lambda a, b: a + ", " + b, object_list)
        object_detections = self._detect_objects(image, additional_tags)
        if filter:
            return self._filter(object_detections)
        return object_detections
    
    def _detect_objects(self, image, additional_tags):
        return self.driver.send_query(image, additional_tags)