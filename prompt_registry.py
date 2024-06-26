import re
import yaml
import inspect

from enum import IntEnum
from open_scene_graph import OSGSpec
from typing import Optional

### Objects to format prompts for queries ###
class PromptType(IntEnum):
    LABEL_PLACE = 0
    APPEARANCE_DESCRIPTION = 1
    SCENE_ELEMENT_CLASSIFICATION = 2
    PLACE_LABEL_SIMILARITY = 3
    PAIRWISE_PLACE_MATCHING = 4

    def __str__(self):
        return self.name

class PromptRegistry:
    """
    Instantiates templated prompts given an OSG spec.

    Also provides utilities to validate and format the
    responses to prompts received from an LLM/VLM in a
    standard format that can be programmatically parsed.
    """

    def __init__(self, spec: type[OSGSpec]):
        self.spec = spec
        self.methods = {
            PromptType.LABEL_PLACE : (self._makePromptPL, self._handleRespPL),
            PromptType.APPEARANCE_DESCRIPTION: (self._makePromptDesc, self._handleRespDesc),
            PromptType.SCENE_ELEMENT_CLASSIFICATION: (self._makePromptClassify, self._handleRespClassify),
            PromptType.PLACE_LABEL_SIMILARITY: (self._makePromptSim, self._handleRespSim),
            PromptType.PAIRWISE_PLACE_MATCHING: (self._makePromptMatch, self._handleRespMatch),
        }

        with open('utils/template_prompts.yaml') as f:
            self.prompt_templates = yaml.safe_load(f)

    def getPrompt(self, prompt_type: type[PromptType], ctx=None):
        """
        Input: prompt_type, PromptType
               ctx, contextual/scene info needed to populate the prompt,
                    apart from OSG spec info
        Output: prompt, string or list, depending on the prompt type
                handler_fn, function to validate and format the response,
                            may be curried with context if that is needed
        """
        assert prompt_type in self.methods, "Unknown prompt type requested"
        make_prompt_fn, validate_fn = self.methods[prompt_type]
        need_ctx_to_validate = len(inspect.getfullargspec(validate_fn)[0]) > 2

        # Return prompt and a handler function to format and validate 
        return (
            make_prompt_fn(ctx),
            lambda r: validate_fn(r, ctx) if need_ctx_to_validate else validate_fn
        )
        
    def _makePromptPL(self, ctx=None):
        """
        Input: ctx, None, no context needed for this function
        """
        temp = self.prompt_templates[str(PromptType.LABEL_PLACE)]['query']
        place_classes = self.spec.getLayerClasses(3)

        # TODO: Currently we can only handle having one place class. Need
        # to generalise to arbitrary no. of place classes
        return temp.format(place_class=place_classes[0])

    def _makePromptDesc(self, ctx: list[str]):
        """
        Input: ctx, list of object names to insert into templated prompt
        """
        temp = self.prompt_templates[str(PromptType.APPEARANCE_DESCRIPTION)]['query']
        return [
            t.format(object_name=object_name)
            for object_name in ctx for t in temp            
        ]
        
    def _makePromptClassify(self, ctx: list[str]):
        """
        Input: ctx, list of detections (as text strings) to be sorted into different layers
        """
        temp = self.prompt_templates[str(PromptType.SCENE_ELEMENT_CLASSIFICATION)]['query']
        place_classes = self.spec.getLayerClasses(3)
        connector_classes = self.spec.getLayerClasses(2)
        queried_classes = place_classes + connector_classes + ["object"]
        obj_conn_list = [s + '_' + str(i) for i, s in enumerate(ctx)]

        fewshot = self.prompt_templates[str(PromptType.SCENE_ELEMENT_CLASSIFICATION)]['fewshot']
        query = temp.format(obj_conn_list=obj_conn_list, queried_classes=queried_classes)
        chat = [
            {"role": "system", "content": "You are a helpful assistant."}
        ] + [
            {"role": k.split('_')[0], "content": v} for k, v in fewshot.items()
        ] + [
            {"role": "user", "content": query}
        ]
        return chat

    def _makePromptSim(self, ctx: list[str]):
        """
        Input: ctx, list of names of all places in OSG
        """
        temp = self.prompt_templates[str(PromptType.PLACE_LABEL_SIMILARITY)]['query']
        place_classes = self.spec.getLayerClasses(3)

        # TODO: Currently does not handle specs with more than one place class
        return temp.format(all_place_labels=ctx, place_class=place_classes[0])


    def _makePromptMatch(self, ctx: dict[str, str]):
        """
        Input: ctx, dict containing descriptions of observed and candidate places
        """
        place_classes = self.spec.getLayerClasses(3)
        temp = self.prompt_templates[str(PromptType.PAIRWISE_PLACE_MATCHING)]
        context = temp['ctx']
        fewshot = temp['fewshot']
        query = temp['query']

        # TODO: Currently does not handle specs with more than one place class
        context.format(place_class=place_classes[0])
        query.format(
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


    def _handleRespPL(self, resp: list[str]):
        return [
            s.replace(" ", "") for s in resp # remove spaces
        ]

    def _handleRespDesc(self, resp: list[str], ctx: list[str]):
        """
        Input: resp, list of strings of (multiple) attrs of objects in ctx
               ctx, list of strings of detected objects to be described
        """
        assert len(resp) == len(ctx) * 2, "resp and ctx have unmatched lengths"
        return [
            resp[i*2] + ' ' + resp[i*2+1] + ' ' + object_name
            for i, object_name in enumerate(ctx)
        ]

    def _handleRespClassify(
            self, 
            resp: str, 
            ctx: list[str],
    ):
        # Get classes specified in query
        place_classes = self.spec.getLayerClasses(3)
        connector_classes = self.spec.getLayerClasses(2)
        queried_classes = place_classes + connector_classes + ["object"]

        # Format and listify response
        print(ctx)
        lowered = resp.lower()
        formatted = lowered.replace(' ', '').replace('.', '')
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
                encoded_idx = int(e.split('_')[-1])
                print("?", encoded_idx, len(ctx))
                print("&", ctx[encoded_idx])
                return (
                    encoded_idx < len(ctx) and
                    e == ctx[encoded_idx]
                )
            except:
                return False
        
        if valid:
            observable_classes = connector_classes + ["object"]
            class_indices = [idx for idx, _ in returned_classes]
            class_ranges = list(zip(
                class_indices, class_indices[1:] + [len(itemised)]
            ))
            print(class_ranges)
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
            print(returned_classes, observable_classes, class_ranges, ranges)
            for cls, (lo, hi) in zip(observable_classes, ranges):
                if cls in observable_classes:
                    print(">", cls)
                    print(itemised[lo+1:hi])
                    print([validate_elem_fn(e) for e in itemised[lo+1:hi]])
            class_to_object_map = {
                cls : [
                    int(e.split('_')[-1]) for e in itemised[lo+1:hi]
                    if validate_elem_fn(e)
                ] for cls, (lo, hi)  in zip(observable_classes, ranges)
                if cls in observable_classes
            }

            return class_to_object_map
        return None

    def _handleRespSim(self, resp: str, ctx: list[str]):
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

    def _handleRespMatch(self, resp: str) -> Optional[bool]:
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