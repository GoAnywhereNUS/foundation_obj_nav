import cv2
import PIL as pil
import numpy as np
import rospy
import actionlib
from sensor_msgs.msg import Image
from obj_nav_actions.msg import TriggerImageNavAction, TriggerImageNavGoal
from utils import msg_to_pil, pil_to_msg


class SelectImageController:
    def __init__(self):
        self.tlx, self.tly = -1, -1
        self.brx, self.bry = -1, -1
        self.movex, self.movey = -1, -1
        self.image_crop = None
        self.drawing = False
        self.curr_sensor_msg = None
        self.executing_task = False

        # Set up ROS node and client
        rospy.init_node("image_controller")
        rospy.loginfo("Setting up client...")
        self.client = actionlib.SimpleActionClient('image_nav', TriggerImageNavAction)
        self.client.wait_for_server()
        rospy.loginfo("Client connected to server!")

        self.image_sub = rospy.Subscriber("/rs_mid/color/image_raw", Image, self.image_cb, queue_size=1)

        rospy.loginfo("Start visualiser...")
        cv2.namedWindow("Viz")
        cv2.setMouseCallback("Viz", self.draw_rectangle)

    def image_cb(self, msg):
        self.curr_sensor_msg = msg

    def done_cb(self, terminal_state, result):
        self.executing_task = False
        rospy.loginfo("Task ended!")
        print(result)

    def feedback_cb(self, feedback):
        print(feedback)

    def send_goal(self, img):
        self.executing_task = True

        rospy.loginfo("Sending goal...")
        goal = TriggerImageNavGoal()
        goal.subgoal = pil_to_msg(pil.Image.fromarray(img))
        
        self.client.send_goal(
            goal,
            done_cb=self.done_cb,
            feedback_cb=self.feedback_cb,
        )

    # Mouse callback function to select rectangular region
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.movex, self.movey = x, y
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.tlx, self.tly = x, y
            self.movex, self.movey = x, y
            self.drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.tlx == x or self.tly == y:
                return

            if self.tlx < x: self.tlx, self.brx = self.tlx, x
            else: self.tlx, self.brx = x, self.tlx
            if self.tly < y: self.tly, self.bry = self.tly, y
            else: self.tly, self.bry = y, self.tly

            if self.curr_sensor_msg is not None:
                img = np.asarray(msg_to_pil(self.curr_sensor_msg))
                cv2.namedWindow("Selected")
                cv2.imshow("Selected", img[self.tly:self.bry, self.tlx:self.brx])
                
                cv2.namedWindow("Crop")
                cropped = self.bbox2crop(
                    img, self.tlx, self.tly, self.brx, self.bry
                )
                cv2.imshow("Crop", cropped)

                self.image_crop = cropped
                self.send_goal(self.image_crop)

    # Fit a given bounding box inside an image crop with
    # same aspect ratio as original image
    def bbox2crop(self, img, tlx, tly, brx, bry):
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
    
    def spin(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            curr_im_array = np.asarray(msg_to_pil(self.curr_sensor_msg))
            print(self.drawing, self.movex, self.movey)
            if self.drawing and (self.movex, self.movey) != (-1, -1):
                cv2.rectangle(curr_im_array, (self.tlx, self.tly), (self.movex, self.movey), (0, 255, 0), 2)
            cv2.imshow("Viz", curr_im_array)
            if cv2.waitKey(1) == ord('q'):
                break
            if cv2.waitKey(1) == ord('s'):
                self.image_crop = None
            if cv2.waitKey(1) == ord('p'):
                pass

            rate.sleep()


if __name__ == "__main__":
    controller = SelectImageController()
    controller.spin()
