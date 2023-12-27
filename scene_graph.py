import json
import networkx as nx
from contextlib import suppress

######## Scene graph specs ########
#
# Scene graph specs have the following requirements:
#   1. Specify all the node types present in the graph
#   2. For each node type, specify the edge types and label them semantically
#   3. The scene graph always contains "object" as the lowest level
#   4. The specs should also specify the agent's state representation.
#      This is done in a list, and should reflect the agent's location with
#      respect to the hierarchical structure of the scene graph. E.g.
#      state = [building1, floor2, livingroom2]

default_scene_graph_specs = """
{
    "room": {
        "contains": ["object"],
        "connects to": ["entrance"]
    },
    "entrance": {
        "is near": ["object"],
        "connects to": ["room"]
    },
    "object": {
    },
    "state": ["room"]
}
"""

######## Scene graph implementation ########

class SceneGraph:
    def __init__(self, specs):
        self.scene_graph_specs = json.loads(specs)
        self.state = self.scene_graph_specs["state"]
        del self.scene_graph_specs["state"]

        self.scene_graph = nx.Graph()

    def nodes(self):
        return self.scene_graph.nodes()

    def is_type(self, node_name, node_type):
        """
        Input:
            node_name: string, label for node that will be used to create unique identifier  
            node_type: string, specifies the type of node 
        Return:
            bool: check whether the node is the specific type or not
        """
        return self.scene_graph.nodes()[node_name]['type'] == node_type

    def get_secific_type_nodes(self, node_type = 'room'):
        """
        Return:
            room_lst: list, all the room in the scene graph
        """
        room_lst = [
           item  for item in self.scene_graph.nodes()
            if self.is_type(item, node_type)
        ]
        return room_lst

    def add_node(self, node_type, node_name, node_dict):
        """
        Adds a node and its attribute dictionary to the scene graph,
        and returns a unique identifier string for that node.

        Input:
            node_type: string, specifies the type of node
            node_name: string, label for node that will be used to create unique identifier
            node_dict: dict, attributes associated with node (e.g. images, other language labels etc.)
        
        Return:
            unique_id: string, unique node identifier modified from node_name
        """
        node_dict['type'] = node_type
        unique_id = self.get_unique_name(node_type, node_name)
        self.scene_graph.add_nodes_from([(unique_id, node_dict)])
        return unique_id
    
    def update_node(self, unique_node_id, node_dict):
        raise NotImplementedError

    def get_node_attr(self, node_name):
        return self.scene_graph._node[node_name]

    def get_obj_in_room(self, room_name):
        obj_lst = []
        if self.scene_graph.has_node(room_name):
            obj_lst = [ item for item in self.scene_graph[room_name]
                if self.scene_graph[room_name][item]['relation'] == 'contains'
            ]
        return obj_lst   

    def add_edge(self, src_node, dst_node, edge_type):
        """
        Adds an edge between src and dst nodes

        Input:
            src_node, dst_node: string, unique scene graph identifier specifying the src and dst nodes
            edge_type: string, describes the type of relation the edge represents
        
        Return:
            boolean, True if edge was successfully added otherwise False
        """
        # Check that this type of edge is allowed in the scene graph specs
        src_node_type = self.scene_graph._node[src_node]['type']
        dst_node_type = self.scene_graph._node[dst_node]['type']
        if dst_node_type not in self.scene_graph_specs[src_node_type][edge_type]:
            return False
        
        self.scene_graph.add_edge(src_node, dst_node, relation=edge_type)
        return True
    
    def add_new_state_and_obs(self, state, edges, node_attrs, non_state_nodes):
        """
        Given a new state that the agent has reached, we update
        the scene graph and add in new observations.

        Input:
            state: dict, following the state representation format from the scene
                   graph specs. Each node_type maps to (flag, label), where label
                   is the language label given to the current node, and flag is a
                   Boolean indicating whether this node should be added to the graph.
            edges: dict, consisting of tuples of (label, relation_dict), where 
                   relation_dict contains the incident edges to the node given by label.
                   Only nodes that will be added or updated should be in edges.
            node_attrs: dict, consisting of tuples (label, node_attr_dict).
            non_state_nodes: dict, listing the non-state nodes to add.
        """
        raise NotImplementedError

    def print_scene_graph(self, json_flag=True, pretty=False, skip_object=True):
        """
        Prints the scene graph as a dict or JSON string. It represents the scene graph
        using the standard format below, which is also used for loading graphs.
        Assume that default_scene_graph_specs are used:

        {
            'rooms': [
                'livingroom1': {
                    'contains': [... objects ...],
                    'connects to': [... doors ...]
                }, ...
            ],
            'entrances': [
                'entrance1': {
                    'is near': [... objects ...],
                    'connects to': [... rooms ...]
                }, ...
            ],
            'objects': ['object1', 'object2', ...]
        }

        Input:
            pretty: If True, pretty prints with newlines.
        
        Return:
            scene_graph_dict or scene_graph_string
        """

        sg_dict = {}

        # Populate each node_type
        for node_type in self.scene_graph_specs.keys():
            if node_type == "state":
                # Ignore state representation key
                continue

            if node_type == "object":
                # Only list all the objects without listing their edges
                # since they are the leaf nodes of the scene graph
                if not skip_object:
                    sg_dict["object"] = [
                        n for n, atts in self.scene_graph.nodes(data=True) 
                        if atts['type'] == "object"
                    ]
                continue

            # Otherwise, list all the nodes of this particular node_type
            # along with all their edges
            node_instances = [
                n for n, atts in self.scene_graph.nodes(data=True) 
                if atts['type'] == node_type
            ]
            edge_types = self.scene_graph_specs[node_type].copy()
            if skip_object:
                edge_types.pop('contains',0)

            node_instance_atts = {
                node_inst: {
                    etype: [
                        dst for _, dst, atts in self.scene_graph.edges(node_inst, data=True) 
                        if atts['relation'] == etype
                    ]
                    for etype in edge_types
                }
                for node_inst in node_instances
            }
            sg_dict[node_type] = node_instance_atts

        if json_flag:
            return json.dumps(sg_dict, indent=(2 if pretty else None))
        else:
            return sg_dict

    def print_sub_scene_graph(self, selected_node, json_flag=True, pretty=False, skip_object=True):
        """
        Prints the scene graph as a dict or JSON string. It represents the scene graph
        using the standard format below, which is also used for loading graphs.
        Assume that default_scene_graph_specs are used:

        {
            'rooms': [
                'livingroom1': {
                    'contains': [... objects ...],
                    'connects to': [... doors ...]
                }, ...
            ],
            'entrances': [
                'entrance1': {
                    'is near': [... objects ...],
                    'connects to': [... rooms ...]
                }, ...
            ],
            'objects': ['object1', 'object2', ...]
        }

        Input:
            pretty: If True, pretty prints with newlines.
        
        Return:
            scene_graph_dict or scene_graph_string
        """

        sg_dict = {}
        sub_graph = nx.subgraph(self.scene_graph, nx.node_connected_component(self.scene_graph, selected_node))
        # Populate each node_type
        for node_type in self.scene_graph_specs.keys():
            if node_type == "state":
                # Ignore state representation key
                continue

            if node_type == "object":
                # Only list all the objects without listing their edges
                # since they are the leaf nodes of the scene graph
                if not skip_object:
                    sg_dict["object"] = [
                        n for n, atts in sub_graph.nodes(data=True) 
                        if atts['type'] == "object"
                    ]
                continue

            # Otherwise, list all the nodes of this particular node_type
            # along with all their edges
            node_instances = [
                n for n, atts in sub_graph.nodes(data=True) 
                if atts['type'] == node_type
            ]
            edge_types = self.scene_graph_specs[node_type].copy()
            if skip_object:
                edge_types.pop('contains',0)

            node_instance_atts = {
                node_inst: {
                    etype: [
                        dst for _, dst, atts in sub_graph.edges(node_inst, data=True) 
                        if atts['relation'] == etype
                    ]
                    for etype in edge_types
                }
                for node_inst in node_instances
            }
            sg_dict[node_type] = node_instance_atts

        if json_flag:
            return json.dumps(sg_dict, indent=(2 if pretty else None))
        else:
            return sg_dict


    def load_scene_graph(self):
        # TODO
        raise NotImplementedError

    def count_instances(self, node_type, node_name):
        instances = [
            node for node, atts in self.scene_graph.nodes(data=True) 
            if atts['type'] == node_type and node.split("_")[0] == node_name
        ]
        return len(instances)
        
    def get_unique_name(self, node_type, node_name):
        num_name_instances = self.count_instances(node_type, node_name)
        return node_name + "_" + str(num_name_instances + 1)
    
    def plan_shortest_paths(self, current_node_name, goal_node_name):
        path = nx.shortest_path(self.scene_graph, current_node_name, goal_node_name)
        return path
    
if __name__ == "__main__":
    import numpy as np
    sg = SceneGraph(default_scene_graph_specs)

    # Add living room
    lr1 = sg.add_node("room", "livingroom", {"image": np.random.rand(4, 4)})
    d1 = sg.add_node("entrance", "door", {"image": np.random.rand(4, 4)})
    d2 = sg.add_node("entrance", "door", {"image": np.random.rand(4, 4)})
    obj1 = sg.add_node("object", "lamp", {"image": np.random.rand(4, 4)})
    obj2 = sg.add_node("object", "vase", {"image": np.random.rand(4, 4)})
    obj3 = sg.add_node("object", "tv", {"image": np.random.rand(4, 4)})

    sg.add_edge(lr1, d1, "connects to")
    sg.add_edge(lr1, d2, "connects to")
    # sg.add_edge(d1, lr1, "connects to")

    sg.add_edge(lr1, obj1, "contains")
    sg.add_edge(lr1, obj2, "contains")

    sg.add_edge(d1, obj2, "is near")

    # Add dining room
    dr1 = sg.add_node("room", "diningroom", {"image": np.random.rand(4, 4)})

    sg.add_edge(dr1, d1, "connects to")

    # Print
    print(sg.scene_graph.edges())
    print(sg.print_scene_graph(pretty=True))

    import pdb
    pdb.set_trace()
    print(sg.print_sub_scene_graph(obj3))
    