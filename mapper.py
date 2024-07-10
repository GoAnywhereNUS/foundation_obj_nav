import torch
import torchvision
import numpy as np
import time
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
from prompt_registry import PromptRegistry, Prompts
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
        self.prompt_reg = PromptRegistry(self.spec)

        # Model interfaces
        self.llm = models["llm"]
        self.vqa = models["vqa"]
        self.obdet = models["obdet"]

        # Hyperparams
        # TODO: Read in from config file
        self.pixel_threshold = 100      # pixels
        self.bbox_iou_threshold = 0.1   # IoU

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
                            "class1": [(obs_idx, label, attr, bbox, im_crop, neighbours), ...],
                            "class2": ...
                        }
                    }, ...
                }
                obs_idx is a unique identifier within the current view
                label is an open-set text label of the object
                attr is an open-set text descriptor of the object
                bbox is the bounding box (x0, x1, y0, y1) in the image
                im_crop is the image crop specified by the bounding box
                neighbours is the list of obs_idx of nearby objects/connectors in current view
        """
        print("##### Image parsing")
        t1 = time.time()
        combined_obdet_bboxes, combined_obdet_labels, combined_obdet_crops = [], [], []
        obdet_to_view = []
        views = dict()
        observable_classes = self.spec.getLayerClasses(2) + ["object"]
        has_multiple_place_classes = len(self.spec.getLayerClasses(3)) > 1
        if has_multiple_place_classes:
            place_class_prompt, place_class_validate_fn = self.prompt_reg.getPromptAndHandler(
                Prompts.PlaceClass)

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
            place_prompt, place_validate_fn = self.prompt_reg.getPromptAndHandler(
                Prompts.LabelPlace, place_class)
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

        t2 = time.time()

        combined_obdet_labels = list(map(
            generic_string_format, combined_obdet_labels))
            
        # Classify objects and connectors in a map that looks like:
        # {"Place1": [...], "Connector1": [...], "Connector2": [...], ...}
        classify_prompt, classify_handle_resp_fn = self.prompt_reg.getPromptAndHandler(
            Prompts.SceneElementClassification, combined_obdet_labels)
        valid, classify_resp = self.llm.query(
            classify_prompt,
            classify_handle_resp_fn,
            required_samples=1,
            max_tries=10,
        )
        if valid:
            class_to_object_map, = classify_resp
        else:
            return None
        
        # TODO: Hack to filter out floors
        for c in class_to_object_map:
            updated_idxs = [
                idx for idx in class_to_object_map[c]
                if not "floor" in combined_obdet_labels[idx]
            ]
            class_to_object_map[c] = updated_idxs
   
        t3 = time.time()

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

        desc_prompts, desc_handle_resp_fn = self.prompt_reg.getPromptAndHandler(
            Prompts.AppearanceDescription, 
            (class_to_object_map, combined_obdet_labels)
        )
        
        print("***** DBG:", len(parsed_ims), len(parsed_labels), len(parsed_bboxes))
        parsed_attrs = [] if len(parsed_ims) == 0 else self.vqa.query(
            parsed_ims, desc_prompts, desc_handle_resp_fn, prompts_per_image=2)

        # Return parsed observations sorted by class, for each view
        for att, lbl, bb, im, cls, view in zip(
            parsed_attrs, parsed_labels, parsed_bboxes, 
            parsed_ims, filtered_elem2class, filtered_obdet2view
        ):
            views[view]["objects"][cls].append((lbl, att, bb, im))

        t4 = time.time()

        # For each view, compute the neighbours for each final detection
        for view in views:
            objects = views[view]["objects"]
            flattened_bboxes = [
                bbox for objs in objects.values() for _, _, bbox, _ in objs
            ]
            nidxs = self._getNeighboursInImage(flattened_bboxes)

            obs_idx = 0
            for cls, objs in views[view]["objects"].items():
                cls_objs = []
                for lbl, att, bbox, im in objs:
                    cls_objs.append((obs_idx, lbl, att, bbox, im, nidxs[obs_idx]))
                    obs_idx += 1
                views[view]["objects"][cls] = cls_objs

        t5 = time.time()
        print("***** Timing:", t2 - t1, t3 - t2, t4 - t3, t5 - t4)

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
        print("##### State estimation")
        t1 = time.time()

        # Get label of current observed Place
        class_and_label = [
            (data["place_class"], data["place"]) for _, data in views.items()
        ]
        consensus_place_class, consensus_place_label = Counter(
            class_and_label).most_common(1)[0][0]

        # Get textual description of current observed Place with object features
        if prev_state is not None:
            object_feats_per_view = {
                view : [
                    lbl for cls_list in data["objects"].values()
                    for _, lbl, _, _, _, _ in cls_list
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
            t2 = time.time()
            print("***** Timing:", t2 - t1)
            
            # Pairwise Place matching
            for idx in sorted_idxs:
                t3 = time.time()
                if np.isinf(shortest_path_lengths[idx]):
                    break
                place_node = similar_place_nodes[idx]
                object_feats = self.OSG.getNodeObjectFeatures(place_node)
                place_desc = self._makePlaceDescription({"forward": object_feats})
                match_prompt_ctx['candidate'] = place_desc
                match_prompt, match_handle_resp_fn = self.prompt_reg.getPromptAndHandler(
                    Prompts.PairwisePlaceMatching, match_prompt_ctx)
                valid, match_resp = self.llm.query(
                    match_prompt,
                    match_handle_resp_fn,
                    required_samples=5,
                )
                t4 = time.time()
                print("***** Timing:", t4 - t3)
                print("***** Pairwise Place Match:", place_node, valid, match_resp)
                if (
                    valid and
                    Counter(match_resp).most_common(1)[0][0] # consensus is a match
                ):
                    return place_node
            
        return (consensus_place_class, consensus_place_label)
        
    def updateOSG(
        self,
        est_state: Union[tuple[str, str], type[OpenSceneGraph.NodeKey]],
        path: list[OpenSceneGraph.NodeKey],
        views,
    ):
        """
        Integrates new observations into the OSG
        """
        print("##### OSG updating")
        print("***** Est. state:", est_state)
        print("***** Path:", path)
        t1 = time.time()

        object_nodes = dict()
        if isinstance(est_state, tuple):
            # Currently in an unrecognisable location. Check against the
            # OSG specification to see if this location can be connected back
            # to the current OSG.
            print("***** In novel Place!")
            place_class, place_label = est_state
            if len(path) > 0:
                last_path_node = path[-1]
                if not self.spec.isConnectable(place_class, last_path_node.node_cls):
                    raise Exception(
                        f"""Invalid spatial structure encountered! {place_class}
                        cannot be connected with {last_path_node.node_cls}."""
                    )
            
            # If current location is valid according to the spec, create a new
            # node and update the spatial connectivity
            est_state = self._addPlaceNode(place_class, (place_label,))
            if len(path) > 0:
                self.OSG.addEdges(est_state, last_path_node, "connects to")

            # Currently in new Place node. Add all observed Objects/Connectors
            # directly as child nodes, and update their connectivity/containment
            # relations wrt current Place node.
            for view, data in views.items():
                nodes, neighbours = [], []

                # For this view, add new leaf nodes and connect them to Place
                for cls, objects in data["objects"].items():
                    added_nodes = [self._addLeafNode(cls, o) for o in objects]
                    nodes += added_nodes
                    neighbours += [nidxs for _, _, _, _ , _, nidxs in objects]

                    if cls in self.spec.getLayerClasses(2): # Connectors
                        self.OSG.addEdges(est_state, added_nodes, "connects to")
                    else: # Objects
                        self.OSG.addEdges(est_state, added_nodes, "contains")

                # For this view, add the spatial proximity edges among leaf nodes
                for node, neighbours in zip(nodes, neighbours):
                    neighbour_nodes = [nodes[idx] for idx in neighbours]
                    self.OSG.addEdges(node, neighbour_nodes, "is near")

                object_nodes[view] = list(zip(
                    nodes,
                    [obj[3] for objs in data["objects"].values() for obj in objs]
                ))
            
            # Update region abstractions
            for _ in range(4, self.spec.getHighestLayerId() + 1):
                pass
        else:
            # Revisiting a previously seen Place in the OSG. Update nodes here.
            assert self.OSG.isPlace(est_state), "Given state is not a Place node!"
            print("***** In existing node", est_state)

            associations = self._associateObservationsWithNodes(views, est_state)
            nodes_neighbours_map = dict()
            for view, assocs in associations.items():
                flattened_objs = [
                    (cls, obj) for cls, objs in views[view]["objects"].items()
                    for obj in objs
                ]

                # For each view, update nodes associated to observations
                # and add unassociated observations as new nodes. Note that
                # updated attribs may be overridden by later views (no fusion
                # of attribs across views currently).
                nodes = []
                for obs_id, is_valid_assoc, assoc_node in assocs:
                    cls, obj = flattened_objs[obs_id]
                    if is_valid_assoc: # obs_id is associated to existing OSG node
                        self._updateLeafNodeAttribs(assoc_node, obj)
                        nodes.append(assoc_node)
                    else: # No valid association to existing node found
                        node = self._addLeafNode(cls, obj)
                        edge_type = (
                            "connects to" if cls in self.spec.getLayerClasses(2)
                            else "contains"
                        )
                        self.OSG.addEdges(est_state, node, edge_type)
                        nodes.append(node)

                # For each view, record the observed neighbours for each
                # added/updated node. (The same node may be updated from
                # multiple views, so we record neighbours across all of them).
                for obs_id, node in enumerate(nodes):
                    _, (_, _, _, _, _, nidxs) = flattened_objs[obs_id]
                    neighbour_nodes = [nodes[idx] for idx in nidxs]
                    if node in nodes_neighbours_map:
                        nodes_neighbours_map[node] += neighbour_nodes
                    else:
                        nodes_neighbours_map[node] = neighbour_nodes

                object_nodes[view] = list(zip(
                    nodes,
                    [obj[3] for _, obj in flattened_objs]
                ))

            # Update spatial proximity edges among the nodes
            for node, neighbours in nodes_neighbours_map.items():
                self.OSG.addEdges(node, neighbours, "is near")
            
        t2 = time.time()
        print("***** Timing:", t2 - t1)
        return est_state, object_nodes
    
    def visualiseOSG(self, est_state):
        self.OSG.visualise(show_layer_1_for_node=est_state)
    
    def _getNeighboursInImage(self, bboxes):
        neighbour_idxs = [[] for _ in range(len(bboxes))]
        if len(bboxes) > 1:
            centroids = torch.Tensor([
                [(x1+x0)*0.5, (y1+y0)*0.5] 
                for x0, y0, x1, y1 in bboxes
            ])
            bboxes = torch.Tensor(bboxes)
            ious = torchvision.ops.box_iou(bboxes, bboxes)
            dists = torch.cdist(centroids, centroids)
            ious_thresholded = ious > self.bbox_iou_threshold
            dists_thresholded = dists < self.pixel_threshold
            neighbours = torch.logical_and(
                ~torch.eye(len(bboxes), dtype=bool),
                torch.logical_or(ious_thresholded, dists_thresholded)
            )
            neighbour_idxs = [
                [idx for idx, flag in enumerate(flags) if flag]
                for flags in neighbours.tolist()
            ]
        return neighbour_idxs

    def _associateObservationsWithNodes(
        self, 
        views, 
        est_state: type[OpenSceneGraph.NodeKey],
    ) -> dict[str, list[tuple]]:
        connectors = self.OSG.getConnectedNodes(est_state)
        objs = self.OSG.getChildNodes(est_state)
        view_associations = dict()

        for view, data in views.items(): # For each view
            associations = []
            flattened_labels = [
                lbl for items in data["objects"].values()
                for _, lbl, _, _, _, _ in items
            ]
            for cls, items in data["objects"].items(): # For each class
                if self.spec.isConnector(cls):
                    existing_nodes = connectors
                else:
                    existing_nodes = None # For now, we don't associate objects

                for obs_id, lbl, _, _, _, nidxs in items: # For each detection
                    valid, classify_resp = False, None
                    if existing_nodes:
                        existing_nodes_and_feats = {
                            str(node): (node, self.OSG.getNodeObjectFeatures(node))
                            for node in existing_nodes
                        }
                        nlabels = [flattened_labels[nidx] for nidx in nidxs]
                        assoc_prompt, assoc_handle_resp_fn = self.prompt_reg.getPromptAndHandler(
                            Prompts.ObjectDataAssociation, ctx={
                                "obs": (lbl, nlabels),
                                "nodes": existing_nodes_and_feats,
                        })
                        valid, classify_resp = self.llm.query(
                            assoc_prompt,
                            assoc_handle_resp_fn,
                            required_samples=5,
                        )
                    associations.append((obs_id, valid, classify_resp))
            view_associations[view] = associations
        return view_associations

    def _addLeafNode(self, cls: str, obs: tuple) -> type[OpenSceneGraph.NodeKey]:
        _, lbl, attr, _, crop, _ = obs
        object_attribs = self.OSG.makeNewNodeAttrs(cls, {
            "label": lbl, "description": attr, "image": crop
        })
        return self.OSG.addNode(cls, object_attribs)
    
    def _addPlaceNode(self, cls: str, obs: tuple) -> type[OpenSceneGraph.NodeKey]:
        lbl, = obs
        place_attribs = self.OSG.makeNewNodeAttrs(cls, {
            "class": cls, "label": lbl,
        })
        return self.OSG.addNode(cls, place_attribs)

    def _updateLeafNodeAttribs(
        self,
        node_key: type[OpenSceneGraph.NodeKey],
        obs: tuple,
    ):
        _, lbl, attr, _, crop, _ = obs
        update_dict = { # Currently only update the attributes and image
            "description": attr,
            "image": crop,
        }
        self.OSG.setNodeAttrs(node_key, update_dict)

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
            curr_heading_feats = (
                "nothing" if len(feats) == 0 
                else reduce(lambda s1, s2: s1 + ", " + s2, feats)
            )
            prep = 'is' if len(curr_heading_feats) <= 1 else 'are'
            if len(object_feats_per_view) == 1:
                heading_desc = f"Around you, there {prep} {curr_heading_feats}."
            else:
                heading_desc = f"In the {heading} direction, there {prep} {curr_heading_feats}."
            place_desc += heading_desc
        return place_desc