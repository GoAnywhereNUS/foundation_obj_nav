import yaml
from collections import deque

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from navigator import *

class NavigatorROS(Navigator):
    def __init__(
        self,
        scene_graph_specs=default_scene_graph_specs,
        llm_config_path="configs/gpt_config.yaml",
        node_config_path="configs/ros_node_config.yaml",
    ):
        
        super().__init__(
            scene_graph_specs=scene_graph_specs, 
            llm_config_path=llm_config_path
        )

        # Initialise ROS node
        with open(node_config_path, 'r') as f:
            self.node_config = yaml.safe_load(f)
        rospy.init_node('navigator', anonymous=True)

        # Data structures
        self.image_buffer = deque(maxlen=5)
        self.bridge = CvBridge()

        # Publishers
        self.image_goal_pub = rospy.Publisher(self.node_config['topics']['image_goal_topic'], Image, queue_size=2)

        # Subscribers
        self.image_sub = rospy.Subscriber(self.node_config['topics']['camera_topic'], Image, self._image_callback)

        # TODO: Services

    def _image_callback(self, image_msg):
        self.image_buffer.append(image_msg)

    def run(self):
        rate = rospy.Rate(self.node_config["params"]["navigator_frequency"])
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

if __name__ == "__main__":
    node = NavigatorROS()
    node.run()