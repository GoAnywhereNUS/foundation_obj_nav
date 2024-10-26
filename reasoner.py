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
        spec,
        action_logger,
        models = {"llm": ModelLLMDriver_GPT}, 
    ):
        self.llm = LLMInterface(models["llm"]())
        self.prompt_reg = PromptRegistry(spec)
        self.action_logging=action_logger

    def reset(self, action_logger):
        self.action_logging = action_logger

    def _proposeRegionSubgoal(
        self, 
        subgraph_string, 
        subgraph_nodes,
        goal, 
        region, 
        explored_nodes
    ):
        region_prompt, region_handle_resp_fn = self.prompt_reg.getPromptAndHandler(
            Prompts.ProposeRegionSubgoal, 
            ctx={
                "scene_graph_layer": subgraph_string,
                "goal": goal,
                "region": region,
                "explored_list": explored_nodes,
                "nodes": { str(node): node for node in subgraph_nodes }
            }
        )
        valid, resp = self.llm.query(
            region_prompt,
            region_handle_resp_fn,
            required_samples=5,
            max_tries=10,
        )

        print("@@@", resp)

        if valid:
            return Counter(resp).most_common(1)[0][0]
        
        # TODO: Should return the current state
        return None
    
    def _proposeObjectSubgoal(self, goal, object_node_list):
        object_prompt, object_handle_resp_fn = self.prompt_reg.getPromptAndHandler(
            Prompts.ProposeObjectSubgoal,
            ctx={
                "object_list": [str(node) for node in object_node_list],
                "goal": goal,
                "nodes": { str(node): node for node in object_node_list }
            }
        )
        valid, resp, raw = self.llm.query(
            object_prompt,
            object_handle_resp_fn,
            required_samples=5,
            max_tries=10,
            get_raw_responses=True
        )

        print("~~~ Propose object subgoal")
        print(raw)
        print(resp)

        if valid:
            return Counter(resp).most_common(1)[0][0]
        
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
        # subgraph = OSG.getLayer(max_layer_id)
        subgraph = OSG.getLayerSubgraph(max_layer_id)

        for layer_id in range(max_layer_id, 2, -1):
            explored_nodes = [k for k in subgraph.nodes() if OSG.isNodeExplored(k) == True]
            subgoal_region = self._proposeRegionSubgoal(
                OSG.printGraph(subgraph=subgraph),
                subgraph.nodes(),
                goal,
                spec.getLayerClasses(layer_id),
                explored_nodes,
            )
            subgraph = OSG.getLayerSubgraph(layer_id)
            print(f"[Subgoal region] {subgoal_region}")

        # Keep only child nodes that are in view, and which also are not explored
        # or cannot be explored (because they are objects)
        child_nodes = [
            node for node in OSG.getChildNodes(subgoal_region)
            if len(OSG.getNode(node)["in_view"]) > 0 and (OSG.isNodeExplored != True)
        ]

        print("======")
        for node in OSG.getChildNodes(subgoal_region):
            print(OSG.getNode(node))

        print(f"[Child nodes]: {child_nodes}")
        self.action_logging.write(f'[Child nodes]: {child_nodes}')

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
        print(f"[Path] {path}")
        print(f"[Subgoal] {subgoal_object}")
        self.action_logging.write(f'[PLAN INFO] Path:{path}\n')
        self.action_logging.write(f'[Next subgoal]: {subgoal_object} | [Last subgoal]: {last_subgoal}')

        return subgoal_object, last_subgoal