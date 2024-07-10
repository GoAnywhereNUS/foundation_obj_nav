import rospy
from PIL import Image
from typing import Optional, Union
from model_interfaces import (
    BaseVQADriver,
    BaseObjectDriver,
    ModelVQADriver,
    ModelObjectDriver,
)
from utils.ros_utils import pil_to_msg, msg_to_pil

from obj_nav_actions.srv import (
    VQAService,
    VQAServiceResponse,
    ObjectService,
    ObjectServiceResponse,
)
from sensor_msgs.msg import Image

######## Driver classes that query models over ROS ########

class ROSVQADriver(BaseVQADriver):
    def __init__(self):
        super().__init__()
        rospy.loginfo("Waiting for VQA service...")
        rospy.wait_for_service('vqa_service')
        self.vqa_service = rospy.ServiceProxy('vqa_service', VQAService)
        rospy.loginfo("Connected to VQA service!")

    def reset(self):
        raise NotImplementedError

    def send_query(
        self, 
        image_input: Union[Image.Image, list], 
        prompt_input: Union[str, list], 
        prompts_per_image: int,
    ):
        if isinstance(image_input, Image.Image):
            image_input = [image_input]
        if isinstance(prompt_input, str):
            prompt_input = [prompt_input]

        image_input = [pil_to_msg(im) for im in image_input]
        
        try:
            response = self.vqa_service(image_input, prompt_input, prompts_per_image)
            return response.answers
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None
        
class ROSObjectDriver(BaseObjectDriver):
    def __init__(self):
        super().__init__()
        rospy.loginfo("Waiting for object service...")
        rospy.wait_for_service('object_service')
        self.object_service = rospy.ServiceProxy('object_service', ObjectService)
        rospy.loginfo("Connected to object service!")

    def reset(self):
        raise NotImplementedError

    def send_query(
        self, 
        image: Image.Image, 
        additional_tags: list[str],
    ):
        try:
            sensor_image = pil_to_msg(image)
            response = self.object_service(sensor_image, additional_tags)
            return response.detected_objects
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None
        
######## Node to run local models ########

class LocalModelNode:
    def __init__(self):
        # ROS node should already have been initialised before,
        # so we issue a warning if the navigator node can be init
        try:
            rospy.init_node("navigator")
            rospy.logwarn("Navigator node not initialised yet! Initialising...")
        except rospy.exceptions.ROSException as e:
            rospy.loginfo("Initialised GNMController in Navigator node.")

        # Load VQA model
        rospy.loginfo("Loading VQA...")
        self.vqa_driver = ModelVQADriver()

        # Load object detection model
        rospy.loginfo("Loading open-set object detector")
        self.object_driver = ModelObjectDriver()

        self.vqa_server = rospy.Service('vqa_service', VQAService, self._handle_vqa_query)
        self.object_server = rospy.Service('object_service', ObjectService, self._handle_object_query)
        rospy.loginfo("Local model services ready!")
        rospy.spin()

    def _handle_vqa_query(self, req):
        image_input = [msg_to_pil(im) for im in req.images]
        resp = self.vqa_driver(
            image_input=image_input, 
            prompt_input=req.prompts, 
            prompts_per_image=req.prompts_per_image
        )
        return VQAServiceResponse(resp)

    def _handle_object_query(self, req):
        image_input = [msg_to_pil(im) for im in req.images]
        resp = self.object_driver(
            image=image_input,
            additional_tags=req.additional_tags,
        )
        return ObjectServiceResponse(resp)