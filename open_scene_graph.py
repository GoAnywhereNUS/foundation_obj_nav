import json
import networkx as nx
import torch

from functools import reduce
from dataclasses import dataclass

######## Scene graph specs ########

default_scene_graph_specs = """
{
    "room": {
        "layer_type": "Place",
        "layer_id": 3,
        "contains": ["Object"],
        "connects to": ["entrance", "room"]
    },
    "entrance": {
        "layer_type": "Connector",
        "layer_id": 2,
        "is near": ["Object"],
        "connects to": ["room"]
    },
    "Object": {
        "layer_type": "Object",
        "layer_id": 1
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

            assert ("contains" in place_cls_spec and "Object" in place_cls_spec["contains"]), \
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
        connectors_layer = layer_map[2]
        places_layer = layer_map[3]
        abstraction_layers = reduce(
            lambda L, k: L if k < 4 else L + [layer_map[k]], 
            layer_map, []
        )

        for connector_cls in connectors_layer:
            connector_cls_spec = spec[connector_cls]
            
            allowed_classes = connectors_layer + ["Object"]
            valid = OSGMetaStructure.checkEdgeSpecValid(connector_cls_spec, "is near", allowed_classes)
            assert valid, f'{connector_cls} has no is near edges or has edges to invalid class'

            allowed_classes = places_layer + abstraction_layers
            valid = OSGMetaStructure.checkEdgeSpecValid(
                connector_cls_spec, "connects to", allowed_classes)
            assert valid, f'{connector_cls} has no connects to edges or has edges to invalid class'

    @staticmethod
    def checkAbstractionLayer(layer_map, spec, i):
        """
        Region abstraction rules:
            1. Must have contains edges to lower Abstractions or Places given in spec
            2. May have connects to edges to Connectors or itself
        """
        connectors_layer = layer_map[2]
        places_layer = layer_map[3]
        lower_abstraction_layers = reduce(
            lambda L, k: L + [layer_map[k]] if k < 4 and k < i else [], 
            layer_map, [])
        
        abstraction_cls, = layer_map[i]
        abstraction_cls_spec = spec[abstraction_cls]

        allowed_classes = places_layer + lower_abstraction_layers
        valid = OSGMetaStructure.checkEdgeSpecValid(abstraction_cls_spec, "contains", allowed_classes)
        assert valid, f'{abstraction_cls} either has no contains edges or edges to an invalid class'
        
        if (
            "connects to" in abstraction_cls_spec 
            and len(abstraction_cls_spec["connects to"]) > 0
        ):
            allowed_classes = connectors_layer + [abstraction_cls]
            valid = OSGMetaStructure.checkValidClasses(
                abstraction_cls_spec["connects to"], allowed_classes)
            assert valid, f'{abstraction_cls} either has no connects to edges or edges to an invalid class'
            
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
    
    def hasEdgeType(self, node_cls: str, edge_type: str):
        return edge_type in self._class_view[node_cls]
        

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

    def getEmptyNodeAttrs(self, node_cls):
        return self.spec.getNodeTemplate(node_cls)

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
    
    def setNodeAttr(self, node_key: type[NodeKey], att_val):
        att, val = att_val
        self.G.nodes()[node_key][att] = val
    
    def getClass(self, node_class: str):
        return [
            node for node, attrs in self.G.nodes(data=True) 
            if attrs["class"] == node_class
        ]
        
    def getLayer(self, layer: int):
        return [
            node for node, attrs in self.G.nodes(data=True)
            if attrs["layer"] == layer
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
            if attrs["label"] == edge_type
        ]

    def getChildNodes(self, node_key: type[NodeKey]):
        return self.getDestNodes(node_key, "contains")

    def getConnectedNodes(self, node_key: type[NodeKey]):
        return self.getDestNodes(node_key, "connects to")
    
    def getNearbyNodes(self, node_key: type[NodeKey]):
        return self.getDestNodes(node_key, "is near")
    
    def getNodeObjectFeatures(self, node_key: type[NodeKey]):
        neighbours = self.getNearbyNodes(node_key)
        return list(map(str, neighbours))
    
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
        edges = [
            (src, dst) for src, dst, data in self.G.edges(data=True)
            if (
                data['label'] == "connects to" and 
                isPlaceConnector(src) and 
                isPlaceConnector(dst)
            )
        ]
        return self.G.edge_subgraph(edges)
    

if __name__ == "__main__":
    graph = OpenSceneGraph(default_scene_graph_specs)
    graph.addNode("Object", {"label": "bed", "description": "hot", "image": torch.rand(4, 4)})
    graph.addNode("entrance", {"label": "door", "description": "wooden", "image": torch.rand(4, 4)})
    graph.addNode("room", {"label": "livingroom", "class": "room"})
    graph.addNode("room", {"label": "livingroom", "class": "room"})

    print([
        node for node, _ in graph.G.nodes(data=True)
    ])