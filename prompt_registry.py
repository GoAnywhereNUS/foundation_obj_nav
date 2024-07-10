import re
import yaml
from typing import Any, Optional, Union, Callable
from functools import reduce

from open_scene_graph import OSGSpec, OpenSceneGraph
from utils.string_utils import generic_string_format

################ Prompt registry and abstract class ################

class BasePrompt:
    def __init__(self, spec, template):
        self.spec = spec
        self.template = template

    def generatePrompt(self, ctx=None):
        raise NotImplementedError("Subclasses should implement this method")
    
    def generateHandler(self, resp: Any, ctx=None):
        raise NotImplementedError("Subclasses should implement this method")
        
class PromptRegistry:
    _registry = {}
    with open('utils/template_prompts.yaml') as f:
        _prompt_templates = yaml.safe_load(f)

    def __init__(self, spec: type[OSGSpec]):
        self.spec = spec

    @classmethod
    def register(cls, prompt_class: type[BasePrompt]):
        name = prompt_class.__name__
        if name in cls._registry:
            raise ValueError(f'Class {name} is already registered!')
        cls._registry[name] = prompt_class

    def getPromptAndHandler(
        self, 
        prompt_class: Union[str, type[BasePrompt]],
        ctx=None,
    ) -> tuple[Any, Callable]:
        """
        Input: prompt_class, i.e. type of prompt specified as str or as class
               ctx, contextual/scene info needed to populate the prompt,
                    apart from OSG spec info
        Output: prompt, string or list, depending on the prompt type
                handler_fn, function to validate and format the response,
                            may be curried with context if that is needed
        """
        if not isinstance(prompt_class, str):
            prompt_class = prompt_class.__name__
        if prompt_class not in self._registry:
            raise ValueError(f'Request prompt {prompt_class} not registered!')
        
        prompt_object = self._registry[prompt_class](
            self.spec,
            self._prompt_templates[prompt_class]
        )
        prompt = prompt_object.generatePrompt(ctx)
        handler = lambda r: prompt_object.generateHandler(r, ctx)
        return prompt, handler

def register_prompt(prompt_class):
    PromptRegistry.register(prompt_class)
    return prompt_class

################ Actual prompt implementations ################

class Prompts: # "Prompts" namespace in which to implement prompts

    @register_prompt
    class PlaceClass(BasePrompt):
        def generatePrompt(self, ctx=None):
            """
            Input: ctx, None, no context needed for this function
            """
            query = self.template['query']
            place_classes = self.spec.getLayerClasses(3)
            return query.format(place_classes=place_classes)
    
        def generateHandler(self, ctx=None):
            raise NotImplementedError
        
    @register_prompt
    class LabelPlace(BasePrompt):
        def generatePrompt(self, ctx : str) -> str:
            """
            Input: ctx, string of identified place class
            """
            query = self.template['query']
            return query.format(place_class=ctx)
        
        def generateHandler(self, resp: list[str], ctx=None):
            formatted = [s.replace(" ", "") for s in resp] # remove spaces
            return formatted[0] # should only have one place label in response
        
    @register_prompt
    class AppearanceDescription(BasePrompt):
        def generatePrompt(
            self, 
            ctx: tuple[dict[str, list[int]], list[str]],
            detailed: bool = False,
        ) -> list[str]:
            """
            Input: ctx, tuple of (class_to_object_map, combined_obdet_labels)
                where the map is {"entrance": [1,2,...], "object": [n, n+1, ...]}
                and each list element is an index into combined_obdet_labels
            """
            query = self.template['query']
            class_to_object_map, detailed_labels = ctx
            flattened_map= [
                (k, detailed_labels[idx])
                for k, v in class_to_object_map.items() for idx in v
            ]
            return [
                q.format(object_name=o) if detailed else q.format(object_name=k)
                for k, o in flattened_map for q in query
            ]
    
        def generateHandler(self, resp: list[str], ctx: list[str]) -> list[str]:
            """
            Input: resp, list of strings of (multiple) attrs of objects in ctx
                ctx, tuple of (class_to_object_map, combined_obdet_labels)
                where the map is {"entrance": [1,2,...], "object": [n, n+1, ...]}
                and each list element is an index into combined_obdet_labels
            """
            class_to_object_map, _ = ctx
            object_count = sum(map(len, class_to_object_map.values()))
            assert len(resp) == object_count * 2, "resp and ctx have unmatched lengths"
            return [
                resp[i*2] + ' ' + resp[i*2+1] for i in range(object_count)
            ]
        
    @register_prompt
    class SceneElementClassification(BasePrompt):
        def generatePrompt(self, ctx: list[str]) -> list[dict[str, str]]:
            """
            Input: ctx, list of detections (as text strings) to be sorted into different layers
            """
            place_classes = self.spec.getLayerClasses(3)
            connector_classes = self.spec.getLayerClasses(2)
            queried_classes = place_classes + connector_classes + ["object"]
            obj_conn_list = [s + '_' + str(i) for i, s in enumerate(ctx)]

            fewshot = self.template['fewshot']
            query = self.template['query'].format(
                obj_conn_list=obj_conn_list,
                queried_classes=queried_classes,
            )

            chat = [
                {"role": "system", "content": "You are a helpful assistant."}
            ] + [
                {"role": k.split('_')[0], "content": v} for k, v in fewshot.items()
            ] + [
                {"role": "user", "content": query}
            ]
            return chat
        
        def generateHandler(self, resp: str, ctx: list[str]):
            # Get classes specified in query
            place_classes = self.spec.getLayerClasses(3)
            connector_classes = self.spec.getLayerClasses(2)
            queried_classes = place_classes + connector_classes + ["object"]

            # Format and listify response
            formatted = generic_string_format(resp)
            itemised = [
                elem for elem in re.split('\n|,|:|-', formatted)
                if elem != 'none'
            ]

            # Check that LLM has returned all classes queried
            returned_classes = [
                (idx, e) for idx, e in enumerate(itemised) if e in queried_classes
            ]
            valid = set([e for _, e in returned_classes]) == set(queried_classes)
            
            # Check the validity of each returned element, return a mapping
            # from the queried classes
            def validate_elem_fn(e: str) -> bool:
                try:
                    object_name, encoded_idx = e.split('_')
                    encoded_idx = int(encoded_idx)
                    return (
                        encoded_idx < len(ctx) and
                        object_name == ctx[encoded_idx]
                    )
                except:
                    return False
            
            if valid:
                observable_classes = connector_classes + ["object"]
                class_indices = [idx for idx, _ in returned_classes]
                class_ranges = list(zip(
                    class_indices, class_indices[1:] + [len(itemised)]
                ))
                ranges = [
                    r for (_, cls), r in zip(returned_classes, class_ranges) 
                    if cls in observable_classes
                ]

                # Maps each queried class to corresponding objects sorted by LLM, i.e.
                # {
                #   "Connector1": [idx1, idx2, idx3, ...],
                #   "Connector2": [idx_n, ...]
                #   "Object": [...]
                # }
                # where each idx is an index into the original combined objects list
                class_to_object_map = {
                    cls : [
                        int(e.split('_')[-1]) for e in itemised[lo+1:hi]
                        if validate_elem_fn(e)
                    ] for cls, (lo, hi)  in zip(observable_classes, ranges)
                    if cls in observable_classes
                }

                return class_to_object_map
            return None
        
    @register_prompt
    class PlaceLabelSimilarity(BasePrompt):
        def generatePrompt(self, ctx: list[str]) -> str:
            """
            Input: ctx, list of names of all places in OSG
            """
            query = self.template['query']
            place_classes = self.spec.getLayerClasses(3)

            # TODO: Currently does not handle specs with more than one place class
            return query.format(all_place_labels=ctx, place_class=place_classes[0])
        
        def generateHandler(
            self, 
            resp: str, 
            ctx: list[str],
        ) -> tuple[list[str], list[str]]:
            """
            Input: resp, string response from VQA
                   ctx, list of all place names queried
            """
            place_resp = resp.split("\n")[-1]
            place_list = place_resp.strip("[]").split(",")
            validated = [(p in ctx) for p in place_list]
            valid_places = [p for p, v in zip(place_list, validated) if v]
            invalid_places = [p for p, v in zip(place_list, validated) if not v]
            return valid_places, invalid_places
        
    @register_prompt
    class PairwisePlaceMatching(BasePrompt):
        def generatePrompt(self, ctx: dict[str, str]) -> list[dict[str, str]]:
            """
            Input: ctx, dict containing descriptions of observed and candidate places
            """
            place_classes = self.spec.getLayerClasses(3)

            # TODO: Currently does not handle specs with more than one place class
            context = self.template['ctx'].format(place_class=place_classes[0])
            fewshot = self.template['fewshot']
            query = self.template['query'].format(
                observed_place_description=ctx['obs'],
                candidate_place_description=ctx['candidate'],
                place_class=place_classes[0]
            )
            chat = [
                {"role": "system", "content": context}
            ] + [
                {"role": k.split('_')[0], "content": v} for k, v in fewshot.items()
            ] + [
                {"role": "user", "content": query}
            ]
            return chat
        
        def generateHandler(self, resp: str, ctx=None) -> Optional[bool]:
            """
            Input: resp, string response from LLM
            Output: option, where None indicates an invalid response,
                    otherwise a boolean indicating matching validity
            """
            answer = resp.split("Answer:")[-1].lower()
            if 'true' in answer:
                return True
            elif 'false' in answer:
                return False
            else:
                return None
            
    @register_prompt
    class ObjectDataAssociation(BasePrompt):
        def generatePrompt(self, ctx: dict[str, Any]) -> list[dict[str, str]]:
            """
            Input: dict, with the structure:
                "obs": (obj_str, nearby_objs_strs),
                "nodes": {"node_str": (node_key, nearby_nodes_strs)}
            """
            obs, obs_feats = ctx['obs']
            nodes = ctx['nodes']

            def feats_to_str_fn(obj, feats):
                if len(feats) >= 2:
                    feats = feats[:-2] + [feats[-2] + ' and ' + feats[-1]]
                feats_str = (
                    'nothing' if len(feats) == 0 else
                    reduce(lambda a, b: a + ', ' + b, feats)
                )
                return f'{obj} that is near {feats_str}.'
            
            target_node_with_feats = feats_to_str_fn(obs, obs_feats)
            nodes = [feats_to_str_fn(n, nodes[n][1]) for n in nodes]
            query = self.template['query'].format(
                target_node_with_feats=target_node_with_feats,
                existing_nodes_with_feats=reduce(lambda a, b: a + ' ' + b, nodes)
            )
            fewshot = self.template['fewshot']
            
            chat = [
                {"role": "system", "content": "You are a helpful assistant."}
            ] + [
                {"role": k.split('_')[0], "content": v} for k, v in fewshot.items()
            ] + [
                {"role": "user", "content": query}
            ]
            return chat
        
        def generateHandler(
            self,
            resp: str, 
            ctx: dict[str, Any]
        ) -> Optional[type[OpenSceneGraph.NodeKey]]:
            node_name = resp.split("Answer:")[-1].strip()
            if node_name in ctx['nodes']:
                return ctx['nodes'][node_name][0]
            return None
    

###############################################################
        
if __name__ == "__main__":
    from open_scene_graph import default_scene_graph_specs
    OSG = OpenSceneGraph(default_scene_graph_specs)
    spec = OSG.getSpec()
    reg = PromptRegistry(spec)
    prompt, handler = reg.getPromptAndHandler(Prompts.LabelPlace, "room")
    print(prompt)
    print(handler(["living room"]))