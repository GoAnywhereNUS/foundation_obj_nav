import yaml
import numpy as np
import threading
from collections import deque

import rospy
import sensor_msgs
from utils.ros_utils import msg_to_pil, pil_to_msg

from navigator import *
import cv2
from gnm_controller import GNMController


class NavigatorROS(Navigator):
    def __init__(
        self,
        scene_graph_specs=default_scene_graph_specs,
        llm_config_path="configs/gpt_config.yaml",
        node_config_path="configs/ros_node_config.yaml",
        visualise=True,
    ):
        
        super().__init__(
            scene_graph_specs=scene_graph_specs, 
            llm_config_path=llm_config_path
        )
        with open(node_config_path, 'r') as f:
            self.node_config = yaml.safe_load(f)
        self.visualisation = visualise

        # Data structures and flags
        self.cam_ids = self.node_config['sensors']['camera_ids']
        self.image_buffer = {
            cam_id : deque(maxlen=5)
            for cam_id in self.cam_ids
        }

        # Initialise ROS node and GNM controller
        rospy.init_node('navigator', anonymous=True)
        self._controller = GNMController(
            maintain_aspect=True
        )

        # Subscribers
        self.cam_subs = [
            rospy.Subscriber(
                '/rs_' + cam_id + '/color/image_raw',
                sensor_msgs.msg.Image,
                lambda msg: self._image_callback(cam_id, msg)
            ) for cam_id in self.cam_ids
        ]

        # TODO: Services

    def _image_callback(self, cam_id, image_msg):
        self.image_buffer[cam_id].append(image_msg)

    def reset(self):
        super().reset()
        self._controller.reset()

    def _observe(self):
        """
        Get observations from the environment (e.g. render images for
        an agent in sim, or take images at current location on real robot).
        To be overridden in subclass.

        Return:
            images: dict of images taken at current pose. None is returned
                    if there are no valid images for any of the sensors.
        """
        obs = None
        if any([len(buf) == 0 for buf in self.image_buffer.values()]):
            rospy.logwarn('No images in buffer for at least one sensor!')
        else:
            obs = { 
                cam_id + '_rgb' : np.array(msg_to_pil(buf[-1]))
                for cam_id, buf in self.image_buffer.items() 
            }
        return obs
    
    def _visualise(self):
        return None

    def run(self):
        """
        Runs the full navigator system
        """
        # rate = rospy.Rate(self.node_config["params"]["navigator_frequency"])

        if self.visualisation:
            cv2.namedWindow("Vis")

        # This loop gets observations, and runs both "slow" and "fast"
        # perception-reasoning. Unlike the sim version, control is not
        # handled in this loop, but is executed asynchronously on the
        # GNM server.
        while not rospy.is_shutdown():
            obs = self._observe()
            if self.visualise:
                self._visualise()

            if obs is not None:
                # High-level perception-reasoning. Run when we have
                # paused and are awaiting next instructions.
                if not self._controller.controller_active():
                    subgoal_image, cam_id = self.loop(obs)
                    if subgoal_image is not None:
                        original_image = obs[cam_id]
                        self._controller.set_subgoal_image(
                            subgoal_image,
                            original_image
                        )

                # TODO: Low-level perception-reasoning. Run all the time.

                if self.visualisation:
                    cv2.imshow("Vis", self.vis_image)

            key = cv2.waitKey(50)
            if key == ord('q'):
                break
            # rate.sleep()

    def run_pr(self):
        """
        Runs a debug, manual version of the navigator without GNM
        """
        active = True
        cv2.namedWindow("Vis")

        while not rospy.is_shutdown():
            obs = self._observe()
            if obs is not None:
                if not active:
                    subgoal_image, cam_id = self.loop(obs)
                    active = True

                if self.visualisation:
                    cv2.imshow("Vis", self.vis_image)

            key = cv2.waitKey(50)
            if key == ord('q'):
                break
            elif key == ord('s'):
                active = False

    def run_pc(self):
        """
        Runs a debug, manual version of navigator system without LLM reasoning.
        """
        main_cam = self.cam_ids[0]
        cv2.namedWindow("Vis")
        cv2.setMouseCallback("Vis", self._select_bbox)

        self.clicked_bbox_idx = None
        self.bbox_lock = threading.Lock()
        self.bboxes = None
        self.labels = None
        perceive = True

        while not rospy.is_shutdown():
            obs = self._observe()
            if obs is None:
                continue

            image = obs[main_cam + '_rgb']

            if not self._controller.controller_active():
                if perceive:
                    img_lang_obs = self.perceive(obs)
                    objects = img_lang_obs['object']
                    bboxes, labels, _ = objects[main_cam]

                    bboxes = bboxes.numpy()
                    bbox_sizes = np.array([
                        (x1-x0) * (y1-y0) for x0, y0, x1, y1 in bboxes
                    ])

                    # Sort from smallest to largest, because we want to
                    # select the smallest bounding box into which our
                    # clicked point falls
                    sorted_idxs = np.argsort(bbox_sizes)
                    bboxes = bboxes[sorted_idxs]
                    labels = np.array(labels)[sorted_idxs]
                    with self.bbox_lock:
                        self.bboxes = bboxes
                        self.labels = labels
                    perceive = False
                else:
                    with self.bbox_lock:
                        if self.clicked_bbox_idx is not None:
                            x0, y0, x1, y1 = self.bboxes[self.clicked_bbox_idx]
                            self._controller.set_subgoal_image(
                                (x0, y0, x1, y1), image
                            )
                            self.clicked_bbox_idx = None
                            
                            cv2.namedWindow("Selection")
                            cv2.imshow("Selection", image[int(y0):int(y1), int(x0):int(x1)])
                            cv2.namedWindow("Recropped")
                            cv2.imshow("Recropped", self._controller.curr_goal)

                for bbox, label in zip(self.bboxes, self.labels):
                    x0, y0, x1, y1 = bbox
                    cv2.rectangle(
                        image, 
                        (int(x0), int(y0)), 
                        (int(x1), int(y1)), 
                        color=(0,0,0), 
                        thickness=1
                    )
                    cv2.putText(
                        image,
                        label,
                        (int(x0), int(y0) - 10),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.6,
                        color = (255, 255, 255),
                        thickness=2
                    )
            else:
                perceive = True

            cv2.imshow("Vis", image)
            key = cv2.waitKey(50)

            if key == ord('p'):
                perceive = True
            elif key == ord('q'):
                return
            elif key == ord('c'):
                cv2.destroyWindow("Selection")
                cv2.destroyWindow("Recropped")
                self._controller.cancel_goal()
                rospy.logwarn("Cancelled goal!")
                

    def _select_bbox(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            # Check to see which bounding box has been selected
            with self.bbox_lock:
                if self.bboxes is not None:
                    for i, (x0, y0, x1, y1) in enumerate(self.bboxes):
                        if x0 < x and x < x1 and y0 < y and y < y1:
                            self.clicked_bbox_idx = i
                            rospy.loginfo(
                                "Selected box " 
                                + str(i) 
                                + ": " 
                                + self.labels[self.clicked_bbox_idx]
                            )
                            return

if __name__ == "__main__":
    node = NavigatorROS()
    node.run_pc()
