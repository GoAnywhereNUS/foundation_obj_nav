import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mapper import OSGMapper

class Navigator:
    def __init__(
        self,
        scene_graph_specs="", 
        llm_config_path="",
        visualise=True,
    ):
        # TODO: Review these variables and parameters.
        self.defined_entrance = ['door', 'doorway', 'doorframe', 'window']
        self.mapper = OSGMapper()
        self.path = []

    def reset(self):
        raise NotImplementedError

    def loop(self, obs):
        print(">>>>> Decision-making")
        print("Goal:", self.goal, "| Path", self.path)
        parsed = self.mapper.parseImage(obs)
        prev_state = None if len(self.path) == 0 else self.path[-1]
        state = self.mapper.estimateState(prev_state, parsed)
        state, object_nodes = self.mapper.updateOSG(state, self.path, parsed)
        self.mapper.visualiseOSG(state)

        if len(self.path) == 0 or state != self.path[-1]:
            self.path.append(state)
        
        while True:
            try:
                selected_goal = input('Please provide the selected goal as <view> <bbox_id>, e.g. forward 2:\n')
                print(f"Got: {selected_goal}")
                if selected_goal.strip().lower() == "q":
                    import sys; sys.exit(0)
                
                view, bbox_id = selected_goal.split(' ')
                bbox_id = int(bbox_id)
                valid = (view in object_nodes) and len(object_nodes[view]) > bbox_id
                if valid:
                    break
            except:
                pass

        selected_node, selected_bbox = object_nodes[view][bbox_id]
        if self.mapper.OSG.isConnector(selected_node):
            self.path.append(selected_node) # Hack to add connectors
        selected_bbox = tuple(map(lambda x: int(np.floor(x)), selected_bbox))
        print(selected_node, selected_bbox, view)
        print(">>>>>")
        self.last_subgoal = self.path
        return selected_bbox, view