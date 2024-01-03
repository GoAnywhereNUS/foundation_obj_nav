import numpy as np
import cv2
import skimage.morphology
from sklearn.cluster import AgglomerativeClustering

# import sys
# sys.path.insert(0, '/home/zhanxin/Desktop/home-robot/src/home_robot')

from home_robot.navigation_planner.fmm_planner import FMMPlanner
from home_robot.core.interfaces import (
    DiscreteNavigationAction,
)

# Select mapping based on what semantic detector you are using. If using the groun truth detector use LABEL_MAP

LABEL_MAP = {
    0: "chair",
    1: "couch",
    2: "potted plant",
    3: "bed",
    4: "toilet",
    5: "tv",
    6: "table",
    7: "oven",
    8: "sink",
    9: "refrigerator",
    10: "book",
    11: "clock",
    12: "vase",
    13: "cup",
    14: "bottle",
    15: "door"
}

REDNET_LABEL_MAP = [
        "chair",
        "sofa",
        "plant",
        "bed",
        "toilet",
        "tv_monitor",
        "bathtub",
        "shower",
        "fireplace",
        "appliances",
        "towel",
        "sink",
        "chest_of_drawers",
        "table",
        "stairs"
]

DETECTRON2_LABEL_MAP = [
        "chair",         # 0
        "sofa",          # 1
        "plant",         # 2
        "bed",           # 3
        "toilet",        # 4
        "tv_monitor",    # 5
        "table",         # 6
        "oven",          # 7
        "sink",          # 8
        "refrigerator",  # 9
        "book",          # 10
        "clock",         # 11
        "vase",          # 12
        "cup",           # 13
        "bottle",        # 14
]


def filter_semantic_map(semantic_map, min_size=20):
    """
    Removes small objects from the semantic map
    """
    filtered_semantic_map = np.zeros_like(semantic_map) + 15
    for c in np.unique(semantic_map):
        if c == 15:
            continue
        semantic_map_c = semantic_map.copy()
        semantic_map_c[semantic_map != c] = 0
        semantic_map_c[semantic_map == c] = 1
        filtered_image = skimage.morphology.remove_small_objects(semantic_map_c.astype(bool), min_size=min_size, connectivity=2).astype(int)
        filtered_semantic_map[filtered_image.astype(bool)] = c
    return filtered_semantic_map


def get_object_info(semantic_map):
    """
    Returns a list of unique objects in the semantic map by grouping pixels with the same label into clusters
    """
    semantic_map_labelled = skimage.measure.label(semantic_map, connectivity=2, background=15)
    props = skimage.measure.regionprops_table(semantic_map_labelled, properties=["centroid", "label", "area"])
    object_cluster_centers = list(zip(props['centroid-0'], props['centroid-1']))
    object_cluster_labels = props['label']
    object_cluster_semantic_labels = []
    for center in object_cluster_centers:
        object_cluster_semantic_labels.append(semantic_map[int(center[0]), int(center[1])])
    return semantic_map_labelled, object_cluster_centers, object_cluster_labels, np.array(object_cluster_semantic_labels)

def dialte_and_compute_distances(object_cluster_centers, traversible_map, semantic_map_labelled, object_cluster_labels, object_cluster_semantic_labels, dialation=10, invalid_thresh=1000):
    """
    For each cluster get a distance map computed by FMMPlanner
    Dialate the goal object so that it is reachable by the robot even if it is on top of or surrounded by a detected obstacle
    """
    # Throws away some labels if the distance map is invalid
    object_cluster_distance_maps, dialated_objects, new_labels, new_semanitc_labels = [], [], [], []
    fmmp = FMMPlanner(traversible_map)
    for center in object_cluster_centers:
        idx = object_cluster_centers.index(center)
        points = semantic_map_labelled == object_cluster_labels[idx]
        # points = np.pad(points, 1, mode='constant', constant_values=0)
        points = skimage.morphology.dilation(points, skimage.morphology.disk(dialation))
        dd = fmmp.set_multi_goal(points.astype(int))
        # detect if valid
        if np.sum(dd == 1) > invalid_thresh:
            print("Invalid distance map", np.sum(dd == 1))
            continue
        object_cluster_distance_maps.append(dd)
        dialated_objects.append(points)
        new_labels.append(object_cluster_labels[idx])
        new_semanitc_labels.append(object_cluster_semantic_labels[idx])
    if len(object_cluster_distance_maps) == 0:
        print("No valid distance maps")
        return None, None, None, None
    return np.stack(object_cluster_distance_maps), np.stack(dialated_objects), np.array(new_labels), np.array(new_semanitc_labels)

def compute_min_object_distances(dialated_objects, stacked_distance_maps):
    """
    Computes the minimum distance from each object to every other object
    """
    object_distances = np.zeros((len(dialated_objects), len(dialated_objects)))
    for i, dialated_object in enumerate(dialated_objects):
        object_distances[i] = np.min((dialated_object * stacked_distance_maps) + ~dialated_object * 100000, axis=(1,2))
    return object_distances

def cluster_object_distances(object_distances, distance_threshold=30):
    """
    Clusters the objects based on the distance between them
    Distances should be computed using compute_min_object_distances
    """
    clustering = AgglomerativeClustering(distance_threshold=distance_threshold, linkage="complete", metric="precomputed", n_clusters=None).fit(object_distances)
    return clustering.labels_

def assign_frontier_to_cluser(labels, frontier_map, distances_maps, max_distance=40):
    """
    Assigns each frontier pixel to the closest object cluster
    If the closest object cluster is more than max_distance away, the frontier pixel is assigned to no cluster
    """
    # Clusters are shifted by 1, 0 indicates no cluster assignment
    distances = np.zeros((len(np.unique(labels)), *frontier_map.shape))
    for i, label in enumerate(np.unique(labels)):
        indices = np.argwhere(labels == label)
        distances[i] = np.min(distances_maps[indices], axis=0) * frontier_map
    extended_distances = np.concatenate([np.zeros((1, *frontier_map.shape)) + frontier_map * max_distance, distances], axis=0)
    return np.argmin(extended_distances, axis=0)

def list_objects_in_clusters(labels, semantic_labels, mapping=REDNET_LABEL_MAP):
    """
    Returns a dictionary of cluster labels to a list of objects in that cluster
    """
    # Print all the semantic ids in each cluster object_cluster_semantic_labels
    semantic_cluster_labels = {}
    if mapping == REDNET_LABEL_MAP:
        unknown = 15
    elif mapping == LABEL_MAP:
        unknown = 16
    else:
        raise NotImplementedError
    for i, label in enumerate(np.unique(labels)):
        objs = semantic_labels[labels == label]
        # print('objs', objs)
        l =  [mapping[obj] for obj in objs if obj != unknown]
        # Remove duplicates 
        l = list(set(l))
        # Sort in alphabetical order
        l.sort()
        semantic_cluster_labels[i] = l
    return semantic_cluster_labels


class DiscreteRecovery:
    def __init__(self, turning_action):
        self.prev_map_pose = None
        self.current_mode = "turn"
        self.max_num_turn_steps = 35
        self.turning_action = turning_action

        # State tracking
        self.perturbation_actions = [DiscreteNavigationAction.MOVE_FORWARD, DiscreteNavigationAction.MOVE_FORWARD]
        self.num_turn_steps = 0

    def get_recovery_action(self, pose):
        if self.prev_map_pose is None:
            self.prev_map_pose = pose
            self.num_turn_steps += 1
            action = "turn_left"
            done = False

            return action, done, self.current_mode

        px, py, ptheta = pose
        prev_x, prev_y, prev_theta = self.prev_map_pose

        # Finite state machine for recovery. Basically:
        # turn -> probe -> perturb -> done
        #  |  ^____|  |      ^__|
        #  V          V
        # failed     failed
        if self.current_mode == "turn":
            # Check effects of turn action (Note: currently ignore effects
            # because we assume turn always succeeds. Directly switch to probe.)
            self.current_mode = (
                "probe" if self.num_turn_steps < self.max_num_turn_steps
                else "failed"
            )
        
        elif self.current_mode == "probe":
            # Check effects of probing
            probe_success = abs(prev_x - px) > 0.05 or abs(prev_y - py) > 0.05
            self.current_mode = "perturb" if probe_success else "turn"

        elif self.current_mode == "perturb":
            self.current_mode = "perturb" if len(self.perturbation_actions) > 0 else "done"

        elif self.current_mode == "done":
            pass

        elif self.current_mode == "failed":
            pass

        else:
            raise NotImplementedError
        
        # Update flags, action to take based on current mode
        if self.current_mode == "turn":
            action, done = self.turning_action, False
            print(">>> Recovery:", action)
            self.num_turn_steps += 1
        elif self.current_mode == "probe":
            print(">>> Recovery: PROBE FORWARD")
            action, done = "move_forward", False
        elif self.current_mode == "perturb":
            action, done = self.perturbation_actions[0], False
            print(">>> Recovery: PERTURB", action)
            self.perturbation_actions.pop(0)
        elif self.current_mode == "done" or self.current_mode == "failed":
            print(">>> Recovery: TERMINATING (", self.current_mode, ")")
            action, done = None, True

        return action, done, self.current_mode