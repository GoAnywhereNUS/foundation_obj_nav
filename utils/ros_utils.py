import PIL as pil
import numpy as np
from sensor_msgs.msg import Image


def msg_to_pil(msg):
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = pil.Image.fromarray(img)
    return pil_image


def pil_to_msg(pil_img, encoding="mono8"):
    img = np.asarray(pil_img)  
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes() 
    ros_image.step = ros_image.width
    return ros_image