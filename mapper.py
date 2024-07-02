import numpy as np
from functools import reduce
from collections import Counter
from typing import Any, Optional, Union
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
from utils.string_utils import *
from utils.logging_utils import draw_annotated_obs


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
        logging=True,
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

        # Debug
        self.logging = {
            "obs": True,
            "state": False,
            "update": False,
        }

    def parseImage(self, obs: dict[str, dict[str, Any]]):
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
        has_multiple_place_classes = len(self.spec.getLayerClasses(3)) > 1
        if has_multiple_place_classes:
            place_class_prompt, place_class_validate_fn = self.prompts.getPrompt(
                PromptType.PLACE_CLASS
            )

        for view, data in obs.items():
            if view == "info":
                continue
            img = Image.fromarray(data['image'])

            # Do Place labelling using entire image
            if has_multiple_place_classes:
                place_class = self.vqa.query(
                    img, place_class_prompt, place_class_validate_fn)
            else:
                place_class = self.spec.getLayerClasses(3)[0]
            place_prompt, place_validate_fn = self.prompts.getPrompt(
                PromptType.LABEL_PLACE, place_class)
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
                "place_class": place_class,
                "place": place_label,
                "objects": {cls : [] for cls in observable_classes},
            }

        combined_obdet_labels = list(map(
            generic_string_format, combined_obdet_labels))
            
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
        parsed_ims, parsed_labels, parsed_bboxes = [], [], []
        filtered_elem2class, filtered_obdet2view = [], []
        for c, idxs in class_to_object_map.items():
            for idx in idxs:
                parsed_ims.append(combined_obdet_crops[idx])
                parsed_labels.append(combined_obdet_labels[idx])
                parsed_bboxes.append(combined_obdet_bboxes[idx].tolist())
                filtered_elem2class.append(c)
                filtered_obdet2view.append(obdet_to_view[idx])

        desc_prompts, desc_handle_resp_fn = self.prompts.getPrompt(
            PromptType.APPEARANCE_DESCRIPTION, 
            (class_to_object_map, combined_obdet_labels)
        )
        parsed_attrs = self.vqa.query(
            parsed_ims, desc_prompts, desc_handle_resp_fn, prompts_per_image=2)

        # Return parsed observations sorted by class, for each view
        for att, lbl, bb, im, cls, view in zip(
            parsed_attrs, parsed_labels, parsed_bboxes, 
            parsed_ims, filtered_elem2class, filtered_obdet2view
        ):
            views[view]["objects"][cls].append((lbl, att, bb, im))

        if self.logging["obs"]:
            draw_annotated_obs(views, obs)

        return views

    def estimateState(
        self,
        prev_state: Optional[type[OpenSceneGraph.NodeKey]],
        views,       
    ) -> Union[str, type[OpenSceneGraph.NodeKey]]:
        """
        Input: prev_state, last Place the agent was located in
               views, processed output from parseImage()
        Output: curr_state, predicted Place in OSG currently occupied by agent,
                            or None if no matching Place is found in OSG
        """
        # Get label of current observed Place
        class_and_label = [
            (data["place_class"], data["place"]) for _, data in views.items()
        ]
        consensus_place_class, consensus_place_label = Counter([
            class_and_label]).most_common(1)[0][0]

        # Get textual description of current observed Pl,ace with object features
        if prev_state is not None:
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
                consensus_place_label, llm_matching=False)
            shortest_path_lengths = self.OSG.getShortestPathLengths(
                prev_state, similar_place_nodes)
            sorted_idxs = np.argsort(shortest_path_lengths) # increasing dist to last known state 
            
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
                consensus_match = Counter([
                    b for b in match_resp]).most_common(1)[0][0]
                if valid and consensus_match:
                    return place_node
            
        return (consensus_place_class, consensus_place_label)
        
    def updateOSG(
        self,
        est_state: Union[tuple[str, str], type[OpenSceneGraph.NodeKey]],
        views,
    ):
        """
        Integrates new observations into the OSG
        """
        if isinstance(est_state, tuple):
            # Currently in an unseen Place: directly add all observations
            # as new nodes to the OSG
            place_class, place_label = est_state
            attr_vals = self.OSG.makeNewNodeAttrs({
                "class": place_class, "label": place_label,
            })
            est_state = self.OSG.addNode(place_class, attr_vals)


            return

        else:
            # Revisiting a previously seen Place in the OSG. Update nodes here.
            return

    def _getSimilarPlaces(
        self,
        place_label: str,
        llm_matching: bool = False,
    ) -> list[type[OpenSceneGraph.NodeKey]]:
        places = self.OSG.getLayer(3)
        if llm_matching:
            raise NotImplementedError
        else:
            return [p for p in places if place_label == p.label]

    def _makePlaceDescription(
        self,
        object_feats_per_view: dict[str, list[str]],
    ) -> str:
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