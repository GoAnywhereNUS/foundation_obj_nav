import yaml
import openai
import torch
from lavis.models import load_model_and_preprocess

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

class GPTInterface(LLMInterface):
    def __init__(
        self,
        key_path="configs/openai_api_key.yaml",
        config_path="configs/gpt_config.yaml",
        ):

        with open(key_path, 'r') as f:
            key_dict = yaml.safe_load(f)
            self.openai_api_key = key_dict['api_key']
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.client = OpenAI(api_key=self.openai_api_key)
        self.chat = [
            {"role": "system", "content": self.config["setup_message"]}
        ]

    def reset(self):
        self.chat = [
            {"role": "system", "content": self.config["setup_message"]}
        ]

    def query(self, string):
        self.chat.append(
            {"role": "user", "content": string}
        )
        response = self.client.chat.completions.create(
            model=self.config["model_type"],
            messages=self.chat,
            seed=self.config["seed"]
        )
        return response


class VLM_BLIP(VQAPerception):
    def __init__(self):
        self.model, self.image_preprocessors, self.text_preprocessors = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=self.device
        )

    def query(self, image, question_prompt):
        image = image.convert("RGB")
        samples = {
            "image": self.vis_processors["eval"](image).unsqueeze(0).to(self.device),
            "text_input": self.txt_processors["eval"](question_prompt)
        }
        ans = self.blip_model.predict_answers(samples=samples, inference_method="generate")
        return ans[0]
    
    
class VLM_GroundingDino(ObjectPerception):
    def __init__(self):
        pass

    def detect_all_objects(self, image):
        # TODO
        return super().detect_all_objects(image)
    
    def detect_specific_objects(self, image, object_list):
        # TODO
        return super().detect_specific_objects(image, object_list)
