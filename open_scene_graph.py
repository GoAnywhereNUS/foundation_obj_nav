import json
import networkx as nx
import numpy as np
import torch

from functools import reduce
from utils.graph_utils import NodeKey

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


class OSGSpec(OSGMetaStructure):
    def __init__(self, spec):
        """
        Input: spec, JSON object
        """

        # Validate and store the spec in queryable form
        OSGMetaStructure.validate(spec)
        self.class_view = spec
        self.layer_view = OSGMetaStructure.getLayerView(spec)

        # Create node templates (i.e. node atts) by updating
        # the defaults in meta-structure with the additional 
        # attributes in the spec
        self.node_attrs = dict()
        for cls, values in self.class_view.items():
            if cls != "state":
                attrs = OSGMetaStructure.node_templates[values["layer_type"]]
                if "attrs" in values:
                    attrs.update(values["attrs"]) # Not yet implemented in specs
                self.node_attrs[cls] = attrs

    def getNodeTemplate(self, node_class):
        if node_class not in self.node_attrs.keys():
            raise Exception("Invalid node type")
        attrs = {
            attr: attr_type()
            for attr, attr_type in self.node_attrs[node_class].items()
        }
        return attrs
    
    def getClassSpec(self, node_cls):
        return self.class_view[node_cls]
    
    def getLayerClasses(self, layer_id):
        return self.layer_view[layer_id]


class OpenSceneGraph:  
    def __init__(self, spec):
        """
        Input: spec, JSON string
        """
        self.spec = OSGSpec(json.loads(spec))
        self.G = nx.DiGraph()

    def addNode(self, node_cls, attr_vals):
        assert "label" in attr_vals, "Need label to add node!"

        attrs = self.spec.getNodeTemplate(node_cls)
        attrs.update(attr_vals)
        attrs["id"] = self.getUniqueId(node_cls, attrs["label"])
        node_key = NodeKey(node_cls=node_cls, label=attrs["label"], uid=attrs["id"])
        self.G.add_node(node_key, **attrs)

    def getNode(self, node_key: type[NodeKey]):
        return self.G.nodes()[node_key]

    def updateNode(self, orig_key: type[NodeKey], new_key: type[NodeKey]):
        pass
    
    def makeNodeKey(self, label: str, layer: int, uid: int):
        return NodeKey(label=label, uid=uid, layer=layer)
    
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

    def getUniqueId(self, node_cls: str, label: str):
        instances = [
            1 for node, _ in self.G.nodes(data=True)
            if (node.node_cls == node_cls and 
                node.label == label)
        ]
        return sum(instances) + 1 # 1-indexed
    

if __name__ == "__main__":
    graph = OpenSceneGraph(default_scene_graph_specs)
    graph.addNode("Object", {"label": "bed", "description": "hot", "image": torch.rand(4, 4)})
    graph.addNode("entrance", {"label": "door", "description": "wooden", "image": torch.rand(4, 4)})
    graph.addNode("room", {"label": "livingroom", "class": "room"})
    graph.addNode("room", {"label": "livingroom", "class": "room"})

    print([
        node for node, _ in graph.G.nodes(data=True)
    ])