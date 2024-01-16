import cv2
import PIL as pil
import numpy as np
import rospy
import actionlib
from controller_interface import Controller
from sensor_msgs.msg import Image
from obj_nav_actions.msg import TriggerImageNavAction, TriggerImageNavGoal
from utils.ros_utils import msg_to_pil, pil_to_msg


class GNMController(Controller):
    def __init__(
        self,
        maintain_aspect=True
    ):
        super().__init__()

        # ROS node should already have been initialised before,
        # so we issue a warning if the navigator node can be init
        try:
            rospy.init_node("navigator")
            rospy.logwarn("Navigator node not initialised yet! Initialising...")
        except rospy.exceptions.ROSException as e:
            rospy.loginfo("Initialised GNMController in Navigator node.")

        # Set up variables
        self.maintain_aspect = maintain_aspect
        self.executing_task = False
        self.curr_sensor_msg = None
        self.curr_goal = None

        # Setting up image_nav action client
        rospy.loginfo("Setting up client...")
        self.client = actionlib.SimpleActionClient('image_nav', TriggerImageNavAction)
        self.client.wait_for_server()
        rospy.loginfo("Client connected to server!")

    def _done_cb(self, terminal_state, result):
        self.executing_task = False
        rospy.loginfo("Task ended!")
        print(result)

    def _feedback_cb(self, feedback):
        pass
        #print(feedback)

    def _bbox2crop(self, img, tlx, tly, brx, bry):
        """
        Fit a given bounding box inside an image crop with same aspect ratio
        as original image.
        """
        cx = 0.5 * (tlx + brx)
        cy = 0.5 * (tly + bry)
        ih, iw, _ = img.shape
        bboxw, bboxh = brx - tlx, bry - tly
        img_ar = float(iw) / float(ih)
        bbox_ar = float(bboxw) / float(bboxh)

        if img_ar > bbox_ar:
            # Fit bbox's height within the crop
            cropw, croph = img_ar * bboxh, bboxh
        else:
            # Fit bbox's width within the crop
            cropw, croph = bboxw, bboxw / img_ar

        tlx = int(max(0, cx - (0.5 * cropw)))
        brx = int(min(iw-1, cx + (0.5 * cropw)))
        tly = int(max(0, cy - (0.5 * croph)))
        bry = int(min(ih-1, cy + (0.5 * croph)))

        return img[tly:bry, tlx:brx]

    def set_subgoal_image(self, subgoal, obs):
        """
        Sets the controller's current subgoal from an image crop.

        Input:
            subgoal: bounding box of the image crop, i.e. (min_x, min_y, max_x, max_y)
            obs: np.ndarray, image from which the subgoal is cropped
        """

        min_x, min_y, max_x, max_y = subgoal
        if self.maintain_aspect:
            crop = self._bbox2crop(obs, min_x, min_y, max_x, max_y)
        else:
            crop = obs[min_y:max_y, min_x:max_x]
        self.curr_goal = crop

        self.executing_task = True
        rospy.loginfo("Sending subgoal...")
        goal = TriggerImageNavGoal()
        goal.subgoal = pil_to_msg(pil.Image.fromarray(self.curr_goal))

        self.client.send_goal(
            goal,
            done_cb=self._done_cb,
            feedback_cb=self._feedback_cb,
        )

    def reset(self):
        # Reset subgoal
        self.cancel_goal()

        # Reset data structures
        self.curr_goal = None
    
    def cancel_goal(self):
        # Cancel all goals running on the server
        # self.client.cancel_all_goals()
        self.client.cancel_goal()
        self.executing_task = False

    def controller_active(self):
        # Returns True if the controller is executing_task
        return self.executing_task

    def visualise(self, obs):
        # TODO: Implement visualisations
        raise NotImplementedError
