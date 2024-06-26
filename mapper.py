import numpy as np
import typing
from functools import reduce
from collections import Counter
from PIL import Image

from open_scene_graph import (
    OpenSceneGraph,
    default_scene_graph_specs,
)
from model_interfaces import (
    GPTInterfaceRefactor,
    VLM_BLIPRefactor,
    VLM_GroundingDino,
)
from prompt_registry import PromptRegistry, PromptType


class OSGMapper:
    """
    A program sketch of an OSG mapping algorithm.
    Specifies the algorithm and prompts in templated
    form, using placeholder spatial concept from the 
    OSG meta-structure.

    Takes as input an OSG spec, which allows it to be
    fully instantiated as an executable routine with
    concrete prompts to LLMs/VLMs.
    """
    def __init__(
        self, 
        models={
            "llm" : GPTInterfaceRefactor(),
            "vqa" : VLM_BLIPRefactor(),
            "obdet" : VLM_GroundingDino(),
        },
        spec_str=default_scene_graph_specs,
    ):
        """
        Input: spec, JSON string
        """
        self.OSG = OpenSceneGraph(spec_str)
        self.spec = self.OSG.getSpec()
        self.prompts = PromptRegistry(self.spec)

        # Model interfaces
        self.llm = models["llm"]
        self.vqa = models["vqa"]
        self.obdet = models["obdet"]

    def parseImage(self, obs: dict[str, dict[str, typing.Any]]):
        """
        Input: obs, nested dictionary where first layer is views, second
                    layer contains observation info. This can be processed
                    (e.g. semantic masks, bounding boxes if available etc.),
                    or can be raw images.
                    Example: {view1: {masks: ..., bboxes: ..., img: ...}}
        Output: parsed observations in a matching form to input
                {
                    "view": { 
                        "place" : ...,
                        "objects" : {
                            "class1": [(label, attr, bbox, im_crop), ...],
                            "class2": ...
                        }
                    }, ...
                }
        """
        combined_obdet_bboxes, combined_obdet_labels, combined_obdet_crops = [], [], []
        obdet_to_view = []
        views = dict()
        observable_classes = self.spec.getLayerClasses(2) + ["object"]
        place_prompt, place_validate_fn = self.prompts.getPrompt(
            PromptType.LABEL_PLACE)

        for view, data in obs.items():
            if view == "info":
                continue
            img = Image.fromarray(data['image'])

            # Do Place labelling using entire image
            place_label = self.vqa.query(img, place_prompt, place_validate_fn)
            
            # Do open-set object detection on entire image
            # Object detections: (bbox_list, label_list, image_crop_list)
            objects = self.obdet.detect_all_objects(img, filter=False)

            # Update detections into combined data structure
            combined_obdet_bboxes += objects[0]
            combined_obdet_labels += objects[1]
            combined_obdet_crops += objects[2]
            obdet_to_view += [view for _ in range(len(objects[0]))]
            views[view] = {
                "place": place_label,
                "objects": {cls : [] for cls in observable_classes}
            }
            
        # Classify objects and connectors in a map that looks like:
        # {"Place1": [...], "Connector1": [...], "Connector2": [...], ...}
        classify_prompt, classify_handle_resp_fn = self.prompts.getPrompt(
            PromptType.SCENE_ELEMENT_CLASSIFICATION, combined_obdet_labels)
        valid, classify_resp = self.llm.query(
            classify_prompt,
            classify_handle_resp_fn,
            required_samples=1,
            max_tries=5,
        )
        if valid:
            class_to_object_map, = classify_resp
        else:
            return None

        # Label object/connector observations with textual attributes
        parsed_ims = [
            combined_obdet_crops[idx]
            for _, idxs in class_to_object_map.items()
            for idx in idxs
        ]
        desc_prompts, desc_handle_resp_fn = self.prompts.getPrompt(
            PromptType.APPEARANCE_DESCRIPTION, class_to_object_map)
        print(len(desc_prompts), len(parsed_ims))
        print(class_to_object_map)
        parsed_attrs = self.vqa.query(
            parsed_ims, desc_prompts, desc_handle_resp_fn, prompts_per_image=2)

        # Return parsed observations sorted by class, for each view
        parsed_labels = [
            combined_obdet_labels[idx]
            for _, idxs in class_to_object_map.items()
            for idx in idxs
        ]
        parsed_bboxes = [
            combined_obdet_bboxes[idx]
            for _, idxs in class_to_object_map.items()
            for idx in idxs
        ]
        elem_to_class = [c for c, idxs in class_to_object_map.items() for _ in idxs]

        for att, lbl, bb, im, cls, view in zip(
            parsed_attrs, parsed_labels, parsed_bboxes, 
            parsed_ims, elem_to_class, obdet_to_view
        ):
            views[view]["objects"][cls].append((lbl, att, bb, im))

        return views

    def estimateState(self, prev_state, views):
        """
        Input: prev_state, last Place the agent was located in
               views, processed output from parseImage()
        Output: curr_state, predicted Place in OSG currently occupied by agent,
                            or None if no matching Place is found in OSG
        """
        # Get label of current observed Place, and textual description of it
        # with object features
        consensus_observed_place = Counter([
            data["place"] for _, data in views.items()
        ]).most_common(1)[0][0]
        object_feats_per_view = {
            view : [
                lbl for cls_list in data["objects"].values()
                for lbl, _, _, _ in cls_list
            ] for view, data in views.items()
        }
        observed_place_desc = self._makePlaceDescription(object_feats_per_view)
        match_prompt_ctx = {'obs': observed_place_desc, 'candidate': None}

        # Get Places that are semantically similar to observed Place
        similar_place_nodes = self._getSimilarPlaces(
            consensus_observed_place, llm_matching=False)
        shortest_path_lengths = self.OSG.getShortestPathLengths(
            consensus_observed_place, similar_place_nodes)
        sorted_idxs = np.argsort(shortest_path_lengths)    
        
        # Pairwise Place matching
        for idx in sorted_idxs:
            if np.isinf(shortest_path_lengths[idx]):
                break
            place_node = similar_place_nodes[idx]
            object_feats = self.OSG.getNodeObjectFeatures(place_node)
            place_desc = self._makePlaceDescription({"forward": object_feats})
            match_prompt_ctx['candidate'] = place_desc
            match_prompt, match_handle_resp_fn = self.prompts.getPrompt(
                PromptType.PAIRWISE_PLACE_MATCHING, match_prompt_ctx)
            valid, match_resp = self.llm.query(
                match_prompt,
                match_handle_resp_fn,
                required_samples=5,
            )
            consensus_match = Counter([b for b in match_resp]).most_common(1)[0][0]
            if valid and consensus_match:
                return place_node
            
        return None
        
    def updateOSG(self):
        pass

    def _getSimilarPlaces(
        self,
        place: type[OpenSceneGraph.NodeKey],
        llm_matching: bool = False,
    ):
        places = self.OSG.getLayer(3)
        if llm_matching:
            raise NotImplementedError
        else:
            return [
                p for p in places if place.label == p.label
            ]

    def _makePlaceDescription(
        self,
        object_feats_per_view: dict[str, list[str]],
    ):
        place_desc = ""
        for heading, feats in object_feats_per_view.items():
            curr_heading_feats = reduce(
                lambda s1, s2: s1 + ", " + s2,
                feats
            )
            prep = 'is' if len(curr_heading_feats) <= 1 else 'are'
            if len(object_feats_per_view) == 1:
                heading_desc = f"Around you, there {prep} {curr_heading_feats}."
            else:
                heading_desc = f"In the {heading} direction, there {prep} {curr_heading_feats}."
            place_desc += heading_desc
        return place_desc