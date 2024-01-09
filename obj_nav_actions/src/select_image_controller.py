import cv2
import PIL as pil
import numpy as np
import threading
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
        self.drawing = False
        self.curr_sensor_msg = None
        self.executing_task = False
        self.goal_lock = threading.Lock()
        self.new_goal_flag = False
        self.image_selection = None
        self.image_recropped =  None

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
        rospy.loginfo("Goal sent!")
        return

    def cancel_goal(self):
        rospy.loginfo("Sending cancel request!")
        self.client.cancel_goal()
        self.image_selection = None
        self.image_recropped = None
        cv2.destroyWindow("Selection")
        cv2.destroyWindow("Recropped")

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
                #cv2.namedWindow("Selected")
                #cv2.imshow("Selected", img[self.tly:self.bry, self.tlx:self.brx])
                
                cropped = self.bbox2crop(
                    img, self.tlx, self.tly, self.brx, self.bry
                )

                #cv2.namedWindow("Crop")
                #cv2.imshow("Crop", cropped)

                with self.goal_lock:
                    self.new_goal_flag = True
                    self.image_selection = img[self.tly:self.bry, self.tlx:self.brx]
                    self.image_recropped = cropped

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
        #rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            with self.goal_lock:
                if self.new_goal_flag:
                    self.send_goal(self.image_recropped)
                    self.new_goal_flag = False

                    cv2.namedWindow("Selection")
                    cv2.imshow("Selection", self.image_selection)
                    cv2.namedWindow("Recropped")
                    cv2.imshow("Recropped", self.image_recropped)
                    
                    #continue

            if self.curr_sensor_msg is not None:
                curr_im_array = np.asarray(msg_to_pil(self.curr_sensor_msg))
                #print(self.drawing, self.movex, self.movey)
                if self.drawing and (self.movex, self.movey) != (-1, -1):
                    cv2.rectangle(curr_im_array, (self.tlx, self.tly), (self.movex, self.movey), (0, 255, 0), 2)
                cv2.imshow("Viz", curr_im_array)

                key = cv2.waitKey(50)
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    pass
                elif key == ord('c'):
                    self.cancel_goal()
            else:
                rospy.logwarn("Waiting for image messages...")

            #rate.sleep()


if __name__ == "__main__":
    controller = SelectImageController()
    controller.spin()
