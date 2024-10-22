from collections import Counter
from open_scene_graph import (
    OpenSceneGraph,
    default_scene_graph_specs,
)
from model_interfaces import (
    LLMInterface,
    ModelLLMDriver_GPT,
)
from prompt_registry import PromptRegistry, Prompts

class Reasoner:
    def __init__(
        self, 
        models = {"llm": ModelLLMDriver_GPT}, 
        action_log_path = "action.txt",
    ):
        self.llm = LLMInterface(models["llm"]())
        self.prompt_reg = PromptRegistry(self.spec)
        self.action_log_path = action_log_path
        self.action_logging = open(action_log_path, 'a')
        self.explored_list = []

    def reset(self, action_log_path):
        self.action_log_path = action_log_path
        self.action_logging = open(self.action_log_path, 'a')

    def _proposeRegionSubgoal(self, subgraph, goal, region):
        region_prompt, region_handle_resp_fn = self.prompt_reg.getPromptAndHandler(
            Prompts.ProposeRegionSubgoal, 
            ctx={
                "scene_graph_layer": subgraph, # TODO: Convert to string
                "goal": goal,
                "region": region,
                "explored_list": self.explored_list, # TODO: Generate explored list
            }
        )
        valid, resp = self.llm.query(
            region_prompt,
            region_handle_resp_fn,
            required_samples=5,
            max_tries=10,
        )

        if valid and Counter(resp).most_common(1)[0][0]:
            return resp
        
        # TODO: Should return the current state
        return None
    
    def _proposeObjectSubgoal(self, goal, object_list):
        object_prompt, object_handle_resp_fn = self.prompt_reg.getPromptAndHandler(
            Prompts.ProposeObjectSubgoal,
            ctx={
                "object_list": object_list,
                "goal": goal
            }
        )
        valid, resp = self.llm.query(
            object_prompt,
            object_handle_resp_fn,
            required_samples=5,
            max_tries=10,
        )

        if valid and Counter(resp).most_common(1)[0][0]:
            return resp
        
        # TODO: Check this
        return None
    
    def _planPath(self, current_node, region_subgoal, spatial_subgraph):
        try:
            if spatial_subgraph.has_path(current_node, region_subgoal):
                path = self.scene_graph.plan_shortest_paths(current_node, region_subgoal)
            else:
                path = [current_node]
        except:
            path = [current_node]
            self.action_logging.write(f'[ERROR] Cannot Find a path between {current_node} and {region_subgoal}')
        return path
        
    def generateExplorationPlan(self, goal, curr_state, OSG):
        spec = OSG.getSpec()
        max_layer_id = spec.getHighestLayerId()
        subgraph = OSG.getLayer(max_layer_id)

        for layer_id in range(max_layer_id, 2, -1):
            subgoal_region = self._proposeRegionSubgoal(
                subgraph,
                goal,
                spec.getLayerClasses(layer_id)
            )
            child_nodes = OSG.getChildNodes(subgoal_region)
            subgraph = OSG.getSubgraphFromNodes(child_nodes)

        # Filter out child nodes that are not in view
        child_nodes = [node for node in child_nodes if OSG.getNode(node)["in_view"]]
        print(f"[Child nodes]: {child_nodes}")
        # self.action_logging.write(f'[Child nodes]: {child_nodes}')

        # Plan a path to the proposed region subgoal
        path = self._planPath(curr_state, subgoal_region, OSG.getSpatialSubgraph())

        # Ground the navigation to an observable object or connector node
        if len(path) == 1 or path[-1] == curr_state:
            # In target room; directly explore for the goal
            subgoal_object = self._proposeObjectSubgoal(goal, child_nodes)
        else:
            # Not yet in target room; explore to reach the next region specified in the path
            subgoal_object = self._proposeObjectSubgoal(path[1], child_nodes)

        last_subgoal = path[0]
        self.action_logging.write(f'[PLAN INFO] Path:{path}\n')
        self.action_logging.write(f'[Next subgoal]: {subgoal_object} | [Last subgoal]: {last_subgoal}')

        return subgoal_object, last_subgoal