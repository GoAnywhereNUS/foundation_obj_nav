'''
Class for VLM Blip and GroundingDINO (GDINO):
'''

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import os
import time

# -----------------------------------------------------------------
# IMPORT for GROUNDING DINO

import numpy as np
import json
import torch
import torchvision
from PIL import Image
# Recognize Anything Model & Tag2Text
import sys
sys.path.append('/home/zhanxin/Desktop/Grounded-Segment-Anything/Tag2Text')
sys.path.append('/home/zhanxin/Desktop/Grounded-Segment-Anything')
sys.path.append('/home/zhanxin/Desktop/Grounded-Segment-Anything/GroundingDINO/')

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


from Tag2Text.models import tag2text
import torchvision.transforms as TS
from Tag2Text.models import tag2text
from Tag2Text import inference_ram
import torchvision.transforms as TS


# -----------------------------------------------------------------



class VLM_BLIP:
    def __init__(self):
        self.device = torch.device("cuda")
        self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=self.device)

    def recoganize_obs(self, obs_rgb, question):
        '''
        Input:
            obs_rgb: PIL image type, eg: obs_rgb = PIL.Image.open("./test.png")
            question: String, eg: "Where is the photo taken?"
        
        Return:
            ans[0]: string
        '''
        # if obs_rgb is array:
        # preprocess_img = Image.fromarray(obs_rgb).convert("RGB")
        # else:
        preprocess_img = obs_rgb.convert("RGB")
        # preprocess_img.save("./output.png", format="PNG")
        image = self.vis_processors["eval"](preprocess_img).unsqueeze(0).to(self.device)
        question = self.txt_processors["eval"](question)
        samples = {"image": image, "text_input": question}
        ans = self.blip_model.predict_answers(samples=samples, inference_method="generate")
        return ans[0]

class VLM_GDINO:
    def __init__(self):
        self.config_file = '/home/zhanxin/Desktop/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'  # change the path of the model config file
        self.ram_checkpoint = '/home/zhanxin/Desktop/mount/ram_swin_large_14m.pth'  # change the path of the model
        self.grounded_checkpoint = "/home/zhanxin/Desktop/mount/groundingdino_swint_ogc.pth"  # change the path of the model
        self.tag2text_checkpoint = '/home/zhanxin/Desktop/mount/tag2text_swin_14m.pth'
        self.sam_hq_checkpoint = None
        self.use_sam_hq = False
        self.split = ","
        self.openai_key = None
        self.openai_proxy = None
        self.box_threshold = 0.25
        self.text_threshold = 0.2
        self.iou_threshold = 0.5
        self.DEFINED_TAGS  = ', doorway, entrance, exit, corridor'
        self.device = torch.device("cuda")


        args = SLConfig.fromfile(self.config_file)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cuda")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        self.model = model

        ram_model = tag2text.ram(pretrained=self.ram_checkpoint, image_size=384, vit='swin_l')
        # threshold for tagging
        # we reduce the threshold to obtain more tags
        ram_model.eval()
        self.ram_model = ram_model.to(self.device)

    def load_image(self, image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def load_obs(self, raw_obs_rgb):
        # load image
        image_pil = Image.fromarray(raw_obs_rgb).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        obs_rgb, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, obs_rgb

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, device="cuda"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        selected_labels = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            selected_labels.append(pred_phrase)
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases, selected_labels



    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
        ax.text(x0, y0, label, ha="left",
                va="top",
                )


    def save_mask_data(self, output_dir, tags_chinese, mask_list, box_list, label_list):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        json_data = {
            'tags_chinese': tags_chinese,
            'mask':[{
                'value': value,
                'label': 'background'
            }]
        }
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data['mask'].append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        with open(os.path.join(output_dir, 'label.json'), 'w') as f:
            json.dump(json_data, f)
        
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

    def Calculate_Nearby_bb(self, target_bbox, all_bbox, all_labels, distance_threshold = 500, overlap_threshold = 0.1):
        #Sample data: Bounding boxes as (x_min, y_min, x_max, y_max)

        # Calculate the center of the specific bounding box
        specific_center = ((target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2)

        # # Find nearby bounding boxes and labels
        nearby_bboxes = []
        nearby_labels = []
        for i in range(len(all_bbox)):
            bbox = all_bbox[i]
            if list(bbox) == list(target_bbox):
                continue
            if np.sqrt((bbox[0] - specific_center[0])**2 + (bbox[1] - specific_center[1])**2) <= distance_threshold:
                nearby_bboxes.append(bbox)
                nearby_labels.append(all_labels[i])
            elif calculate_iou(specific_bbox, bbox) > overlap_threshold:
                nearby_bboxes.append(bbox)
                nearby_labels.append(all_labels[i])
        return nearby_bboxes, nearby_labels

    def object_detect(self, current_obs_rgb, save_path= None, save_idx=None):
        
        # os.makedirs(output_dir, exist_ok=True)

        # load obs
        image_pil, image = self.load_image(current_obs_rgb)
        
        # initialize Recognize Anything Model
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = TS.Compose([
                        TS.Resize((384, 384)),
                        TS.ToTensor(), normalize
                    ])
        
        # load model
        raw_image = image_pil.resize((384, 384))
        raw_image  = transform(raw_image).unsqueeze(0).to(self.device)

        res = inference_ram.inference(raw_image , self.ram_model)

        tags=res[0].replace(' |', ',')
        # tags_chinese=res[1].replace(' |', ',')

        # tags += DEFINED_TAGS
        
        # print("Image Tags: ", res[0])
        # print("Update Image Tags: ", tags)

        # run grounding dino model

        boxes_filt, scores, pred_phrases, selected_labels = self.get_grounding_output(
            self.model, image, tags, self.box_threshold, self.text_threshold, device='cuda'
        )
        
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        selected_labels = [selected_labels[idx] for idx in nms_idx]
        return boxes_filt, selected_labels


'''
TEST VLM Blip and GroundingDINO (GDINO):
'''

# blip_model = VLM_BLIP()
# raw_image = Image.open("/home/zhanxin/Desktop/mount/EnvTest/test.png")
# output = blip_model.recoganize_obs(raw_image, "Where is the photo taken?")
# print(output)


# GDINO_model = VLM_GDINO()
# boxes, labels = GDINO_model.object_detect("/home/zhanxin/Desktop/mount/EnvTest/test.png")
# print('Object:', labels, boxes)
