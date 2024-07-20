import json
import networkx as nx
import torch
import matplotlib.pyplot as plt

from functools import reduce
from dataclasses import dataclass
from typing import Any, Union

######## Scene graph specs ########

default_scene_graph_specs = """
{
    "room": {
        "layer_type": "Place",
        "layer_id": 3,
        "contains": ["object"],
        "connects to": ["entrance", "room"]
    },
    "entrance": {
        "layer_type": "Connector",
        "layer_id": 2,
        "is near": ["object"],
        "connects to": ["room"]
    },
    "object": {
        "layer_type": "Object",
        "layer_id": 1,
        "is near": ["object", "entrance"]
    },
    "state": ["room"]
}
"""

############### Meta-structure #############

class OSGMetaStructure:
    """
    Meta-structure describes the minimal data and allowable
    structure of an OSG.

    Meta-structure class exposes validation functions that encode
    the "rules" of an OSG structure, and check that a given OSG
    spec obeys these rules.

    Meta-structure class also provides node templates with the
    minimum required attributes, which can be extended in an
    OSG structure.
    """

    node_templates = {
        'Object': {'label': str, 'id': int, 'description': str, 'image': torch.Tensor},
        'Connector': {'label': str, 'id': int, 'description': str, 'image': torch.Tensor},
        'Place': {'class': str, 'label': str, 'id': int},
        'Region Abstraction': {'label': str, 'id': int},
    }
        
    @staticmethod
    def validate(spec):
        """
        Validates that an OSG spec follows the rules on
        OSG structure laid out in meta-structure.
        
        spec: OSG specification as JSON object
        """
        # Sort according to layer
        layer_map = OSGMetaStructure.getLayerView(spec)
        
        # Check required layers
        OSGMetaStructure.ensureRequiredLayers(layer_map)
        OSGMetaStructure.checkPlacesLayer(layer_map, spec)

        # Check optional layers
        if 2 in layer_map:
            OSGMetaStructure.checkConnectorsLayer(layer_map, spec)
        highest_idx = max(layer_map.keys())
        for i in range(4, highest_idx + 1):
            assert i in layer_map, f'Unspecified layer index {i} in region abstraction'
            OSGMetaStructure.checkAbstractionLayer(layer_map, spec, i)
            
    @staticmethod
    def ensureRequiredLayers(layer_map):
        """
        Ensures that layers required by the meta-structure are present
        """
        assert 1 in layer_map, "Missing required Objects layer"
        assert 3 in layer_map, "Missing required Places layer"

    @staticmethod
    def checkPlacesLayer(layer_map, spec):
        """
        Places rules:
            1. Must have contains edges, and minimally should contain Objects
            2. Must have connects to edges, to Places/Connectors given in spec
        """
        places_layer = layer_map[3]
        connectors_layer = layer_map[2] if 2 in layer_map else []

        for place_cls in places_layer:
            place_cls_spec = spec[place_cls]

            assert ("contains" in place_cls_spec and "object" in place_cls_spec["contains"]), \
                f'{place_cls} does not contain Objects'
            
            allowed_classes = places_layer + connectors_layer
            valid = OSGMetaStructure.checkEdgeSpecValid(
                place_cls_spec, "connects to", allowed_classes)
            assert valid, f'{place_cls} has no connects to edges or has edges to invalid class'
                
    @staticmethod
    def checkConnectorsLayer(layer_map, spec):
        """
        Connectors rules:
            1. Must have "is near" edges to Connectors/Objects given in spec
            2. Must have "connects to" edges to Places/Abstractions given in spec
        """
        connectors_layer = layer_map[2] if 2 in layer_map else []
        places_layer = layer_map[3]
        abstraction_layers = reduce(
            lambda L, k: L if k < 4 else L + [layer_map[k]], 
            layer_map, []
        )

        for connector_cls in connectors_layer:
            connector_cls_spec = spec[connector_cls]
            
            allowed_classes = connectors_layer + ["object"]
            valid = OSGMetaStructure.checkEdgeSpecValid(connector_cls_spec, "is near", allowed_classes)
            assert valid, f'{connector_cls} is not near any valid items'

            allowed_classes = places_layer + abstraction_layers
            valid = OSGMetaStructure.checkEdgeSpecValid(
                connector_cls_spec, "connects to", allowed_classes)
            assert valid, f'{connector_cls} does not connect to any valid place layer'

    @staticmethod
    def checkAbstractionLayer(layer_map, spec, i):
        """
        Region abstraction rules:
            1. Must have contains edges to lower Abstractions or Places given in spec
            2. May have connects to edges to Connectors or itself
        """
        connectors_layer = layer_map[2] if 2 in layer_map else []
        places_layer = layer_map[3]
        lower_abstraction_layers = reduce(
            lambda L, k: L + [layer_map[k]] if k < 4 and k < i else [], 
            layer_map, [])
        
        abstraction_cls, = layer_map[i]
        abstraction_cls_spec = spec[abstraction_cls]

        allowed_classes = places_layer + lower_abstraction_layers
        valid = OSGMetaStructure.checkEdgeSpecValid(abstraction_cls_spec, "contains", allowed_classes)
        assert valid, f'{abstraction_cls} does not contain any valid place layer.'
        
        if (
            "connects to" in abstraction_cls_spec 
            and len(abstraction_cls_spec["connects to"]) > 0
        ):
            allowed_classes = connectors_layer + [abstraction_cls]
            valid = OSGMetaStructure.checkValidClasses(
                abstraction_cls_spec["connects to"], allowed_classes)
            assert valid, f'{abstraction_cls} does not connect to any valid region.'
            
    @staticmethod
    def checkEdgeSpecValid(spec, edge_type, allowed_classes):
        if edge_type in spec:
            return OSGMetaStructure.checkValidClasses(
                spec[edge_type], allowed_classes)
        return False
    
    @staticmethod
    def checkValidClasses(dst_nodes, allowed_classes):
        valid_classes = [
            dst_node_class in allowed_classes
            for dst_node_class in dst_nodes
        ]
        return len(dst_nodes) > 0 and all(valid_classes)

    @staticmethod
    def getLayerView(spec):
        """
        Generates a layer-first view of the OSG spec,
        which is a dictionary where keys are layer index,
        and values are the various classes in the layer.
        E.g. Layer index 3 (Places) --> [Rooms, Hallways].
        """
        layer_map = dict()
        for k, v in spec.items():
            if k != "state":
                if v["layer_id"] in layer_map.keys():
                    layer_map[v["layer_id"]].append(k)
                else:
                    layer_map[v["layer_id"]] = [k]
        return layer_map

############### OSG Specification #############

class OSGSpec(OSGMetaStructure):
    def __init__(self, spec):
        """
        A schema describing the OSG structure for a 
        particular class of environments, e.g. structure of
        household environments.

        Note that OSGSpec should be treated as an immutable object.

        Input: spec, JSON object
        """
        # Validate and store the spec in queryable form
        OSGMetaStructure.validate(spec)
        self._class_view = spec
        self._layer_view = OSGMetaStructure.getLayerView(spec)

        # Create node templates (i.e. node atts) by updating
        # the defaults in meta-structure with the additional 
        # attributes in the spec
        self._node_attrs = dict()
        for cls, values in self._class_view.items():
            if cls != "state":
                attrs = OSGMetaStructure.node_templates[values["layer_type"]]
                if "attrs" in values:
                    attrs.update(values["attrs"]) # Not yet implemented in specs
                self._node_attrs[cls] = attrs

    def getNodeTemplate(self, node_cls):
        if node_cls not in self._node_attrs.keys():
            raise Exception("Invalid node type")
        attrs = {
            attr: attr_type()
            for attr, attr_type in self._node_attrs[node_cls].items()
        }
        return attrs
    
    def getClassSpec(self, node_cls):
        return self._class_view[node_cls]
    
    def getLayerClasses(self, layer_id):
        return self._layer_view[layer_id]
    
    def getHighestLayerId(self):
        return max(self._layer_view.keys())

    def isConnectable(self, src_node_cls: str, dst_node_cls: str) -> bool:
        """
        Verifies that two classes can be spatially connected according to spec.
        Ensures two-way connectivity.
        """
        src_cls_spec = self._class_view[src_node_cls]
        dst_cls_spec = self._class_view[dst_node_cls]
        return (
            "connects to" in src_cls_spec and
            "connects to" in dst_cls_spec and
            dst_node_cls in self._class_view[src_node_cls]["connects to"] and
            src_node_cls in self._class_view[dst_node_cls]["connects to"]
        )
    
    def isProximable(self, src_node_cls: str, dst_node_cls: str) -> bool:
        """
        Verifies that dst class can be near the src class
        """
        cls_spec = self._class_view[src_node_cls]
        return "is near" in cls_spec and dst_node_cls in cls_spec["is near"]

    def isContainable(self, src_node_cls: str, dst_node_cls: str) -> bool:
        """
        Verifies that dst class can be contained by the src class.
        """
        cls_spec = self._class_view[src_node_cls]
        return "contains" in cls_spec and dst_node_cls in cls_spec["contains"]
        
    def isObject(self, node_cls: str) -> bool:
        return self._class_view[node_cls]["layer_id"] == 1
    
    def isConnector(self, node_cls: str) -> bool:
        return self._class_view[node_cls]["layer_id"] == 2
    
    def isPlace(self, node_cls: str) -> bool:
        return self._class_view[node_cls]["layer_id"] == 3
    
    def isRegionAbstraction(self, node_cls: str) -> bool:
        return self._class_view[node_cls]["layer_id"] > 3

############### OSG #############

class OpenSceneGraph:
    @dataclass(frozen=True)
    class NodeKey:
        node_cls : str
        label : str
        uid : int

        def __repr__(self):
            return f'{self.node_cls}_{self.label}_{self.uid}'
        
        def __str__(self):
            return f'{self.label}_{self.uid}'

    def __init__(self, spec):
        """
        Input: spec, JSON string
        """
        self.spec = OSGSpec(json.loads(spec))
        self.G = nx.DiGraph()

    ### Adding or updating nodes
    def getDefaultNodeAttrs(self, node_cls: str):
        return self.spec.getNodeTemplate(node_cls)
    
    def makeNewNodeAttrs(
        self, 
        node_cls: str, 
        attr_vals: dict[str, Any] = None,
    ):
        default_attrs = self.getDefaultNodeAttrs(node_cls)
        default_attrs.update(attr_vals)
        return default_attrs

    def addNode(self, node_cls, attr_vals):
        assert "label" in attr_vals, "Need label to add node!"

        attrs = self.spec.getNodeTemplate(node_cls)
        attrs.update(attr_vals)
        attrs["id"] = self._getUniqueId(node_cls, attrs["label"])
        node_key = OpenSceneGraph.NodeKey(
            node_cls=node_cls, label=attrs["label"], uid=attrs["id"])
        self.G.add_node(node_key, **attrs)
        return node_key

    def getNode(self, node_key: type[NodeKey]):
        return self.G.nodes()[node_key]
    
    def setNodeAttrs(
        self, 
        node_key: type[NodeKey], 
        atts_vals: dict[str, Any],
    ):
        nx.set_node_attributes(self.G, {node_key: atts_vals})

    ### Adding or updating edges
    def addEdges(
        self, 
        src: type[NodeKey], 
        dsts: Union[type[NodeKey], list[type[NodeKey]]], 
        edge_type: str
    ):
        dsts = dsts if isinstance(dsts, list) else [dsts]
        if edge_type == "contains":
            for dst in dsts:
                self._addContainmentEdge(src, dst)
        elif edge_type == "connects to":
            for dst in dsts:
                self._addConnectivityEdge(src, dst)
        elif edge_type == "is near":
            for dst in dsts:
                self._addProximityEdge(src, dst)
        else:
            raise Exception(f"Tried to add invalid edge type {edge_type}!")
    
    def _addConnectivityEdge(self, src: type[NodeKey], dst: type[NodeKey]):
        if self.spec.isConnectable(src.node_cls, dst.node_cls):
            self.G.add_edges_from([(src, dst), (dst, src)], edge_type="connects to")
        else:
            print(f"Not adding invalid edge. {dst} cannot be contained in {src}!")

    def _addProximityEdge(self, src: type[NodeKey], dst: type[NodeKey]):
        if self.spec.isProximable(src.node_cls, dst.node_cls):
            self.G.add_edge(src, dst, edge_type="is near")
        else:
            print(f"Not adding invalid edge. {dst} cannot be near {src}!")

    def _addContainmentEdge(self, src: type[NodeKey], dst: type[NodeKey]):
        if self.spec.isContainable(src.node_cls, dst.node_cls):
            self.G.add_edge(src, dst, edge_type="contains")
        else:
            print(f"Not adding invalid edge. {dst} cannot be near {src}!")

    def removeEdge(self, src: type[NodeKey], dst: type[NodeKey]):
        self.G.remove_edge(src, dst)

    def resetOutEdges(self, src: type[NodeKey]):
        """
        Removes all out edges from a given src node
        """
        outgoing_edges = list(self.G.out_edges(src))
        self.G.remove_edges_from(outgoing_edges)

    ### Querying data from OSG
    def getClass(self, node_class: str):
        return [
            node for node, attrs in self.G.nodes(data=True) 
            if attrs["class"] == node_class
        ]
        
    def getLayer(self, layer: int):
        layer_classes = self.spec.getLayerClasses(layer)
        return [
            node for node in self.G.nodes()
            if node.node_cls in layer_classes
        ]
    
    def getDestNodes(
        self,
        src_node_key: type[NodeKey],
        edge_type: str,
    ):
        """
        Get all successor nodes that are connected to a given source node
        (src_node_key) by a given edge type (edge_type)
        """
        return [
            dst for _, dst, attrs in self.G.out_edges(src_node_key, data=True)
            if attrs["edge_type"] == edge_type
        ]

    def getChildNodes(self, node_key: type[NodeKey]):
        return self.getDestNodes(node_key, "contains")

    def getConnectedNodes(self, node_key: type[NodeKey]):
        return self.getDestNodes(node_key, "connects to")
    
    def getNearbyNodes(self, node_key: type[NodeKey]):
        return self.getDestNodes(node_key, "is near")
    
    def getNodeObjectFeatures(
        self, 
        node_key: type[NodeKey], 
        return_descriptors: bool = False,
    ):
        if self.isPlace(node_key):
            object_feat_nodes = self.getChildNodes(node_key)
        elif self.isConnector(node_key) or self.isObject(node_key):
            object_feat_nodes = self.getNearbyNodes(node_key)
        else:
            raise Exception(
                "Object features may only be queried for Object/Connector/Place nodes"
            )
        stringified = list(map(str, object_feat_nodes))
        if return_descriptors:
            raise NotImplementedError
        return stringified
    
    def getShortestPathLengths(
        self,
        src_node_key: type[NodeKey],
        dst_node_keys: list[type[NodeKey]],
    ):
        spatial_subgraph_view = self._getSpatialSubgraph()
        lengths = []
        for dst in dst_node_keys:
            try:
                lengths.append(nx.shortest_path_length(
                    spatial_subgraph_view, src_node_key, dst))
            except nx.NetworkXNoPath:
                lengths.append(float('inf'))
        return lengths

    def getSpec(self):
        return self.spec
    
    ### Misc utilities
    def isObject(self, node: type[NodeKey]):
        layer_id = self.spec.getClassSpec(node.node_cls)["layer_id"]
        return layer_id == 1

    def isConnector(self, node: type[NodeKey]):
        layer_id = self.spec.getClassSpec(node.node_cls)["layer_id"]
        return layer_id == 2

    def isPlace(self, node: type[NodeKey]):
        layer_id = self.spec.getClassSpec(node.node_cls)["layer_id"]
        return layer_id == 3

    def isRegion(self, node: type[NodeKey]):
        layer_id = self.spec.getClassSpec(node.node_cls)["layer_id"]
        return layer_id >= 3

    ### Other internal functions for the OSG
    def _getUniqueId(self, node_cls: str, label: str):
        instances = [
            1 for node, _ in self.G.nodes(data=True)
            if (node.node_cls == node_cls and 
                node.label == label)
        ]
        return sum(instances) + 1 # 1-indexed
    
    def _getSpatialSubgraph(self):
        """
        Extracts a subgraph representing the scene's spatial topology from 
        the OSG. Specifically, this is the subgraph of all the Places and
        Connectors joined together with "connects to" edges.
        """
        def isPlaceConnector(n):
            layer_id = self.spec.getClassSpec(n.node_cls)['layer_id']
            return layer_id == 2 or layer_id == 3

        nodes = [node for node in self.G.nodes() if isPlaceConnector(node)]
        return self.G.subgraph(nodes)
    
    ### Visualisation utilities
    def visualise(self, show_layers=[], show_layer_1_for_node=None):
        """
        Visualizes the hierarchical graph.
        
        Parameters:
        - G: NetworkX graph with nodes having the 'layer_id' attribute.
        - show_layers: List of layers to show. If None, only layers 2 and 3 are shown.
        - show_layer_1_for_node: If not None, show Layer 1 nodes that are children of this node.
        """
        edge_colors = {'contains': 'blue', 'connects to': 'green', 'is near': 'red'}
        def get_edge_color(u, v):
            return edge_colors.get(self.G[u][v].get('edge_type', 'type1'), 'black')

        # Create subgraph for layers 2 and 3
        spatial_subgraph = self._getSpatialSubgraph()

        # Extract nodes for each layer
        layer_nodes = {
            i: [n for n, d in self.G.nodes(data=True) if d['id'] == i]
            for i in range(1, self.spec.getHighestLayerId() + 1)
        }

        # Calculate positions for each layer separately
        pos = {}
        layer_height = 1 / (len(layer_nodes) + 1)

        for layer, nodes in layer_nodes.items():
            if not nodes:
                continue
            layer_graph = self.G.subgraph(nodes)
            layer_pos = nx.spring_layout(layer_graph)
            for n in layer_pos:
                layer_pos[n][1] = layer_pos[n][1] * layer_height + (layer_height * (layer - 1))
            pos.update(layer_pos)
        
        plt.figure(figsize=(12, 8))

        # Draw spatial subgraph (layers 2 and 3)
        edge_colors_sub = [get_edge_color(u, v) for u, v in spatial_subgraph.edges()]
        nx.draw(spatial_subgraph, pos, with_labels=True, labels={n: str(n) for n in spatial_subgraph.nodes()}, 
            node_color='lightblue', edge_color=edge_colors_sub, node_size=500, font_size=10)

        # Draw higher layers if toggled on
        for layer in show_layers:
            if layer in layer_nodes:
                layer_graph = self.G.subgraph(layer_nodes[layer])
                layer_edges = layer_graph.edges()
                layer_edge_colors = [get_edge_color(u, v) for u, v in layer_edges]
                nx.draw(layer_graph, pos, with_labels=True, labels={n: str(n) for n in layer_nodes[layer]}, 
                        node_color='orange', edge_color=layer_edge_colors, node_size=700, font_size=12, alpha=0.6)
        
        # Draw Layer 1 nodes if toggled on for a specific node in G_sub
        if show_layer_1_for_node:
            if show_layer_1_for_node in spatial_subgraph:
                layer_1_nodes = list(self.G.successors(show_layer_1_for_node))
                layer_1_graph = self.G.subgraph(layer_1_nodes)
                layer_1_pos = nx.spring_layout(layer_1_graph)
                for n in layer_1_pos:
                    layer_1_pos[n][1] = layer_1_pos[n][1] * layer_height
                    layer_1_pos[n][0] = pos[show_layer_1_for_node][0] + (layer_1_pos[n][0] - 0.5) * 0.2  # Adjust horizontally around parent
                pos.update(layer_1_pos)
                layer_1_edges = layer_1_graph.edges()
                layer_1_edge_colors = [get_edge_color(u, v) for u, v in layer_1_edges]
                nx.draw(layer_1_graph, pos, with_labels=True, labels={n: str(n) for n in layer_1_nodes}, 
                        node_color='green', edge_color=layer_1_edge_colors, node_size=300, font_size=8, alpha=0.8)
        
        plt.title("Hierarchical Graph Visualization")
        plt.savefig("imgs/osg.png")
    

if __name__ == "__main__":
    graph = OpenSceneGraph(default_scene_graph_specs)
    graph.addNode("Object", {"label": "bed", "description": "hot", "image": torch.rand(4, 4)})
    graph.addNode("entrance", {"label": "door", "description": "wooden", "image": torch.rand(4, 4)})
    graph.addNode("room", {"label": "livingroom", "class": "room"})
    graph.addNode("room", {"label": "livingroom", "class": "room"})

    print([
        node for node, _ in graph.G.nodes(data=True)
    ])