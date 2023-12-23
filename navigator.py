from scene_graph import SceneGraph, default_scene_graph_specs
from model_interfaces import GPTInterface, VLM_BLIP, VLM_GroundingDino
import json

class Navigator:
    def __init__(
        self, 
        scene_graph_specs=default_scene_graph_specs,
        llm_config_path="configs/gpt_config.yaml"
    ):

        # Initialise foundation models used for agent's functions
        self.llm = GPTInterface(config_path=llm_config_path)
        self.perception = {
            "object": VLM_GroundingDino(),
            "vqa": VLM_BLIP()
        }

        # Initialise agent's memory (i.e. scene graph, agent's state)
        if scene_graph_specs is None:
            # TODO: Query LLM to get the specs
            raise NotImplementedError
        self.scene_graph_specs = json.loads(scene_graph_specs)
        self.scene_graph = SceneGraph(scene_graph_specs)
        self.state_spec = self.scene_graph_specs["state"]
        self.current_state = {node_type: None for node_type in self.state_spec}
        self.plan = None
        self.last_subgoal = None
        self.is_navigating = False

    def reset(self):
        self.scene_graph = SceneGraph(self.scene_graph_specs)
        self.llm.reset()

    def query_objects(self, image, suggested_objects=None):
        if suggested_objects is not None:
            return self.perception['object'].detect_specific_objects(image, suggested_objects)
        else:
            return self.perception['object'].detect_all_objects(image)

    def query_vqa(self, image, prompt):
        return self.perception['vqa'].query(image, prompt)
    
    def _observe(self):
        """
        Get observations from the environment (e.g. render images for
        an agent in sim, or take images at current location on real robot).
        To be overridden in subclass.

        Return:
            images: dict of images taken at current pose
        """
        raise NotImplementedError

    def perceive(self, images):
        """
        Process raw images into image-language observations for the LLM's
        consumption.

        Return:
            location: name of current location answered by VQA
            objects: list of all objects observed in current vicinity
        """
        image = self._get_image()
        location = self.query_vqa(image, "Where is this photo taken?")
        objects = self.query_objects(image)
        return {
            "location": location,
            "object": objects
        }
    
    def estimate_state(self, obs):
        """
        Queries the LLM with the observations and scene graph to
        get our current state.

        Return:
            state: agent's estimated state as dict in the format
                   {'floor': (new_flag, semantic_label), 'room': (new_flag, semantic_label), ...}
                   where new_flag indicates whether a new node needs to be added to this
                   particular level, and semantic_label is the language label to that node 
                   from the VLM.
        """
        raise NotImplementedError
    
    def update_scene_graph(self, est_state, obs):
        """
        Updates scene graph using localisation estimate from LLM, and
        observations from VLM.

        Return:
            state: agent's current state as dict, e.g. {'floor': xxx, 'room': xxx, ...}
        """

    def plan_path(self):
        # TODO: Query LLM

        path = self.scene_graph.plan_shortest_paths(self.current_state, goal_node_name)[0]
        return path
    
    def has_reached_subgoal(self, state, subgoal):
        raise NotImplementedError
    
    def send_navigation_subgoal(self, subgoal):
        raise NotImplementedError

    def loop(self):
        """
        Single iteration of the navigation loop

        Returns:
            None
        """
        if self.is_navigating:
            return
        
        # Get "observations" from VLMs
        image_lang_obs = self.observe()

        # Localise and update scene graph
        state = self.estimate_state(image_lang_obs)
        state = self.update_scene_graph(state, image_lang_obs)
        self.current_state = state

        # If not currently executing a plan, get new plan from LLM.
        # If currently executing a plan, get next subgoal if we have
        # successfully executed current subgoal, else re-plan with LLM.
        if self.plan is None or not self.has_reached_subgoal(state, self.last_subgoal):
            self.plan = self.plan_path()
        next_subgoal = self.plan[-1]

        self.is_navigating = True
        self.send_navigation_subgoal(next_subgoal)
        self.plan.pop()

    def run(self):
        """
        Executes the navigation loop. To be implemented in each
        Navigator subclass.

        Returns:
            None
        """
        raise NotImplementedError
