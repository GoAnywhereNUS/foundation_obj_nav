import os
import time
import yaml
import numpy as np
import torch
import argparse
from collections import deque

import rospy
import actionlib
from obj_nav_actions.msg import TriggerImageNavAction, TriggerImageNavFeedback, TriggerImageNavResult
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Float32

from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class GNMNode:
    def __init__(
        self,
        args, 
        robot_config_path="../config/spot.yaml", 
        model_config_path="../config/models.yaml", 
    ):
        # Set up params and data structures
        with open(robot_config_path, 'r') as f:
            self.robot_config = yaml.safe_load(f)
        assert self.robot_config is not None

        self.MAX_V = self.robot_config["max_v"]
        self.MAX_W = self.robot_config["max_w"]
        self.RATE = self.robot_config["frame_rate"]
        self.waypoint = args.waypoint
        self.num_samples = args.num_samples
        
        self.curr_sensor_msg = None
        self.recent_dist_buffer = deque(maxlen=4)
        self.curr_filtered_dist = None
        self.min_filtered_dist = np.inf
        self.stop_dist_threshold = 4.0

        # Set up model
        rospy.loginfo("Loading model...")

        with open(model_config_path, "r") as f:
            model_paths = yaml.safe_load(f)
        
        model_config_path = model_paths[args.model]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)
        
        self.model_params["min_linear_vel"] = 0.05
        self.model_params["min_angular_vel"] = 0.03

        ckpth_path = model_paths[args.model]["ckpt_path"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo("Creating model on: " + str(self.device))
        if os.path.exists(ckpth_path):
            print(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
        self.model = load_model(
                ckpth_path, 
                self.model_params, 
                self.device,
        )
        self.model.eval()

        if self.model_params["model_type"] == "nomad":
            self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )

        rospy.loginfo(
            "Loaded model of size: " 
            + str(sum(p.numel() for p in self.model.parameters()))
        )

        # Set up remaining data structures that needed model_params
        self.window_context_queue = deque(maxlen=self.model_params["context_size"] + 1)

        # Set up ROS node
        rospy.init_node('gnm_node')
        rospy.loginfo('Setting up publish/subscribe...')
        self.image_sub = rospy.Subscriber(
            "/rs_mid/color/image_raw", 
            Image, self.image_cb, queue_size=1
        )
        #self.odom_sub = rospy.Subscriber(
        #    "/spot/odometry", 
        #    Odometry, self.odom_cb, queue_size=5
        #)
        self.waypoint_pub = rospy.Publisher("/gnm/waypoint", Float32MultiArray, queue_size=5)
        self.debug_dist_pub = rospy.Publisher("/gnm/dbg_dist", Float32, queue_size=1)
        self.debug_filtered_pub = rospy.Publisher("/gnm/dbg_filt", Float32, queue_size=1)
        self.debug_min_pub = rospy.Publisher("/gnm/dbg_min", Float32, queue_size=1)

        # Set up ROS server
        rospy.loginfo("Setting up server...")
        self.server = actionlib.SimpleActionServer('image_nav', TriggerImageNavAction, self.navigate, False)
        self.server.start()
        rospy.loginfo("Server ready!")

    def image_cb(self, msg):
        self.curr_sensor_msg = msg

    def odom_cb(self, msg):
        # TODO: Implement
        return

    def update_zupt(self):
        pass

    def reached_goal(self):
        # Update distance monitor
        if self.recent_dist_buffer.maxlen == len(self.recent_dist_buffer):
            self.curr_filtered_dist = np.mean(self.recent_dist_buffer)

            if (
                (self.curr_filtered_dist - self.min_filtered_dist)
                > self.stop_dist_threshold
            ):
                return True
            
            self.min_filtered_dist = min(
                self.curr_filtered_dist, 
                self.min_filtered_dist
            )

        return False

    def reset(self):
        self.curr_sensor_msg = None
        self.recent_dist_buffer = deque(maxlen=4)
        self.curr_filtered_dist = None
        self.min_filtered_dist = np.inf
        self.window_context_queue.clear() 

    def navigate(self, subgoal):
        rospy.loginfo("Received subgoal!")
        image_crop = msg_to_pil(subgoal.subgoal)
        rate = rospy.Rate(self.RATE)

        self.test_time = time.time()

        while not rospy.is_shutdown():
            # Check if goal is pre-empted
            if self.server.is_preempt_requested():
                rospy.logwarn("Preempt request received. Cancelling goal...")
                self.server.set_preempted()
                break

            # Get latest observations
            curr_im_pil = msg_to_pil(self.curr_sensor_msg)
            self.window_context_queue.append(curr_im_pil)
            
            # Inference
            context_queue = list(self.window_context_queue)
            if len(context_queue) == self.model_params["context_size"] + 1:

                # NoMaD
                if self.model_params["model_type"] == "nomad":
                    obs_images = transform_images(context_queue, self.model_params["image_size"], center_crop=False)
                    obs_images = torch.cat(torch.split(obs_images, 3, dim=1), dim=1).to(self.device)
                    mask = torch.zeros(1).long().to(self.device)
                    goal_image = transform_images(image_crop, self.model_params["image_size"], center_crop=False).to(self.device)

                    start_time = time.time()
                    obsgoal_cond = self.model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                    dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                    dists = to_numpy(dists.flatten())

                    with torch.no_grad():
                        # encoder vision features
                        if len(obsgoal_cond.shape) == 2:
                            obs_cond = obsgoal_cond.repeat(self.num_samples, 1)
                        else:
                            obs_cond = obsgoal_cond.repeat(self.num_samples, 1, 1)

                        # initialize action from Gaussian noise
                        noisy_action = torch.randn(
                            (self.num_samples, self.model_params["len_traj_pred"], 2), device=self.device)
                        naction = noisy_action

                        # init scheduler
                        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                        for k in self.noise_scheduler.timesteps[:]:
                            # predict noise
                            noise_pred = self.model(
                                'noise_pred_net',
                                sample=naction,
                                timestep=k,
                                global_cond=obs_cond
                            )
                            # inverse diffusion step (remove noise)
                            naction = self.noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=naction
                            ).prev_sample

                    elapsed = time.time() - start_time
                    naction = to_numpy(get_action(naction))
                    #sampled_actions_msg = Float32MultiArray()
                    #sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
                    #print("published sampled actions")
                    #sampled_actions_pub.publish(sampled_actions_msg)
                    naction = naction[0] 
                    chosen_waypoint = naction[self.waypoint]

                # ViNT
                else: 
                    transf_goal_img = transform_images(image_crop, self.model_params["image_size"]).to(self.device)
                    transf_obs_img = transform_images(context_queue, self.model_params["image_size"]).to(self.device)
                    start_time = time.time()
                    dist, waypoint = self.model(transf_obs_img, transf_goal_img) 
                    dist = to_numpy(dist[0])
                    self.recent_dist_buffer.append(dist[0])
                    # dist_queue.append(dist[0])
                    waypoint = to_numpy(waypoint[0])
                    chosen_waypoint = waypoint[self.waypoint]
                    elapsed = time.time() - start_time

                # Check success conditions. If we have reached goal, end task
                # and publish the results. Otherwise publish feedback to the client.
                if self.reached_goal():
                    rospy.loginfo("Reached goal!")
                    self.debug_dist_pub.publish(dist)
                    self.debug_filtered_pub.publish(self.curr_filtered_dist)
                    self.debug_min_pub.publish(self.min_filtered_dist)

                    result = TriggerImageNavResult()
                    self.server.set_succeeded(result)

                    self.reset()
                    return
                
                else:
                    feedback = TriggerImageNavFeedback()
                    self.server.publish_feedback(feedback)

                # Process and publish predicted waypoint
                if self.model_params["normalize"]:
                    chosen_waypoint[:2] *= (self.MAX_V / self.RATE)
                waypoint_msg = Float32MultiArray()
                rospy.loginfo(
                        "Elapsed: " + f"{elapsed:.3f}" 
                    + " | " + str(chosen_waypoint) + ", " + str(dist)
                )
                # TODO: Should we separately scale chosen waypoint with MAX_V and MAX_W?

                waypoint_msg.data = chosen_waypoint
                self.waypoint_pub.publish(waypoint_msg)

                # For debugging
                self.debug_dist_pub.publish(dist)
                self.debug_filtered_pub.publish(self.curr_filtered_dist)
                self.debug_min_pub.publish(self.min_filtered_dist)

            else:
                rospy.loginfo("Filling context queue: " + str(len(context_queue)))

            rate.sleep()

        # If we exit the loop, we have been preempted
        self.reset()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        default="vint",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: large_gnm)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=1,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()

    controller = GNMNode(args)
