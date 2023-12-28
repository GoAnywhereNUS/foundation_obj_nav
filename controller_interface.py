import torch


class Controller:
    def __init__(
        self, 
        device=torch.device('cuda:0')
    ):
        self.curr_goal = None
        self.device = device

    def set_subgoal_coord(self, subgoal, obs=None):
        """
        Set the controller's current subgoal as coordinates in world frame. 
        If needed, the controller can take in current sensor observations 
        to ground the subgoal.
        NOTE: Only used in simulation setup, where we have access to a metric
        coordinate system.
        """
        raise NotImplementedError
    
    def set_subgoal_image(self, subgoal, obs):
        """
        Set the controller's current subgoal as an image crop.
        """
        raise NotImplementedError
    
    def reset(self):
        """
        Clears controller state
        """
        raise NotImplementedError
    
    def reset_subgoal(self):
        """
        Clears the previously set subgoal and resets state.
        """
        raise NotImplementedError
    
    def spin(self):
        """
        ROS spin; controller's main work is done through asynchronous callbacks.
        NOTE: Only used on real robot systems using ROS, where the lower-level
        perception-control loop is asynchronous with higher-level perception-reasoning loop.
        """
        raise NotImplementedError
    
    def step(self):
        """
        Plans a path toward the set goal, and takes a step towards it.
        NOTE: Only used in simulation, where lower-level perception-control 
        loop is synchronous with higher-level perception-reasoning loop.
        """
        raise NotImplementedError
    
    def update(self, obs):
        """
        Update the internal map from observations.
        NOTE: Only used in simulation, which builds a metric grid map for control.
        """
        raise NotImplementedError
    
    def visualise(self, obs):
        """
        Visualise the observations, internal map, pose (and plan?)

        Input:
            obs: home_robot Observation type

        Output:
            image: cv2 Image combining all the above visuals
        """
        raise NotImplementedError