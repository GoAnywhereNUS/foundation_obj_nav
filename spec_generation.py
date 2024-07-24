from open_scene_graph import (
    OpenSceneGraph
)

from model_interfaces import (
    LLMInterface,
    ModelLLMDriver_GPT
)
from prompt_registry import PromptRegistry, Prompts
import networkx as nx

MetaStructure = {'relation':['is near', 'connects to', 'contains']} 

class AutoSpec:
    def __init__(self, env_type):
        """
        Input: String env_type, e.g. home
        """
        self.llm = LLMInterface(ModelLLMDriver_GPT())
        self.max_trial = 5
        self.env_type = env_type
        self.prompt_reg = PromptRegistry(None)
        self.env_ctx = {'environment_type': self.env_type}

    def reset(self):
        self.env_ctx = {'environment_type':self.env_type}

    def getTextEnvLayout(self):
        generate_text_prompt, generate_text_resp_fn = self.prompt_reg.getPromptAndHandler(
            Prompts.TextDescriptionEnvLayout, self.env_ctx)
        valid, generate_resp = self.llm.query(
            generate_text_prompt,
            generate_text_resp_fn,
            required_samples=1,
            max_tries=10,
        )
        self.env_ctx['text_description'] = generate_resp[0]
        self.env_ctx['chat_history'] = generate_text_prompt + [{"role": "assistant", "content": "Text:"+ generate_resp[0]}]
        print('generate_resp', generate_resp)


    def getTriplets(self):
        triplets_text_prompt, triplets_text_resp_fn = self.prompt_reg.getPromptAndHandler(
            Prompts.TextToTriplets, self.env_ctx)
        valid, triplets_resp = self.llm.query(
            triplets_text_prompt,
            triplets_text_resp_fn,
            required_samples=1,
            max_tries=10,
        )
        print('triplets_resp', triplets_resp[0])
        self.env_ctx['environment_triplets'] = triplets_resp[0]

    def find_longest_path(self, graph):
        def dfs(current_node, path, visited):
            nonlocal longest_path
            path.append(current_node)
            visited.add(current_node)

            # Update the longest path if the current path is longer
            if len(path) > len(longest_path):
                longest_path = path.copy()

            # Explore neighbors
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:  # Avoid cycles
                    dfs(neighbor, path, visited)

            # Backtrack
            path.pop()
            visited.remove(current_node)

        longest_path = []
        for node in graph.nodes:
            dfs(node, [], set())

        return longest_path

    def addLayer(self, layer, layer_num, attr_set):
        layer_spec = ""
        layer_type_mapping = {1:"Object", 2:"Connector", 3:"Place", 4: "Region Abstraction", 5: "Region Abstraction", 6: "Region Abstraction", 7: "Region Abstraction"}
        layer_type = f" \"layer_type\": \"{layer_type_mapping[layer_num]}\" "
        layer_id = f" \"layer_id\": {layer_num}"
        
        current_layer_attr = {i: None for i in MetaStructure['relation']}
        if layer_num == 3:
            attr_set['contains'][layer] = ["object"]

        for attr in current_layer_attr.keys():
            if layer in attr_set[attr].keys():
                current_layer_attr[attr] =  f" \"{attr}\": {list(set(attr_set[attr][layer]))}"
        
        if layer_num >= 3:
            current_layer_attr['is near'] = None

        layer_spec += f""" "{layer}": """ + "{\n"
        for each_attr in [layer_type, layer_id, current_layer_attr['contains'], current_layer_attr['is near'], current_layer_attr['connects to']]:
            if each_attr != None:
                layer_spec += f"""    {each_attr},\n"""

        pos = layer_spec.rfind(",\n")
        layer_spec = layer_spec[:pos]
        layer_spec += "\n}," 
        return layer_spec
    
    def Canonicalization(self, triplet_lines):
        refined_triplet = []
        for each_triplet in triplet_lines:
            ctx = {'given_triplet': each_triplet, 'environment_type':self.env_ctx['environment_type']}
            canonicalization_prompt, canonicalization_resp_fn = self.prompt_reg.getPromptAndHandler(
                Prompts.Canonicalization, ctx)
            valid, refined_triplets_resp = self.llm.query(
                canonicalization_prompt,
                canonicalization_resp_fn,
                required_samples=1,
                max_tries=10,
            )
            print('--- before triplets', each_triplet)
            refined_triplet.append(refined_triplets_resp[0])
            print('--- after triplets', refined_triplets_resp[0])
        self.env_ctx['environment_triplets'] = refined_triplet
        return refined_triplet

    def Triplets2Spec(self, triplet_lines):
        hierachy = nx.DiGraph()
        ToBeAddNodes = []

        attr_set = {i: {} for i in MetaStructure['relation']}

        for each_triplet in triplet_lines:
            subject_i = each_triplet[0]
            relation_i = each_triplet[1]
            object_i = each_triplet[2]
            ToBeAddNodes.append(subject_i)
            ToBeAddNodes.append(object_i)

            if relation_i in attr_set.keys():
                if 'contain' in relation_i:
                    hierachy.add_edge(subject_i, object_i)
                    if subject_i not in attr_set['contains'].keys():
                        attr_set['contains'][subject_i] = [object_i]
                    else:
                        attr_set['contains'][subject_i].append(object_i)
                elif 'connect' in relation_i or 'near' in relation_i:
                    if subject_i not in attr_set[relation_i].keys():
                        attr_set[relation_i][subject_i] = [object_i]
                    else:
                        attr_set[relation_i][subject_i].append(object_i)
                    if object_i not in attr_set[relation_i].keys():
                        attr_set[relation_i][object_i] = [subject_i]
                    else:
                        attr_set[relation_i][object_i].append(subject_i)

        ToBeAddNodes = set(ToBeAddNodes)

        longest_path = self.find_longest_path(hierachy)
        if len(longest_path) < 1:
            return None
        if 'object' in longest_path[-1]:
            longest_path.remove(longest_path[-1])
        specs = ""
        layer_num = 3
        layer_spec_list = []
        print('contains', attr_set['contains'])
        print('connects_to', attr_set['connects to'])
        print('is_near',attr_set['is near'])
        print('longest_path', longest_path)
        for layer in longest_path[::-1]:
            if layer in ToBeAddNodes:
                ToBeAddNodes.remove(layer)
                layer_spec = self.addLayer(layer, layer_num, attr_set)
                layer_spec_list.append(layer_spec)
                if layer in attr_set['connects to'].keys():
                    for connected_layer in attr_set['connects to'][layer]:
                        if connected_layer not in ToBeAddNodes:
                            continue
                        if connected_layer in attr_set['is near'].keys() and connected_layer not in attr_set['contains'].keys():
                            layer_spec = self.addLayer(connected_layer, 2, attr_set)
                        else:
                            layer_spec = self.addLayer(connected_layer, layer_num, attr_set)
                        
                        ToBeAddNodes.remove(connected_layer)
                        layer_spec_list.append(layer_spec)
                    
                layer_num += 1

        for each_spec in layer_spec_list[::-1]:
            specs += each_spec
        specs = specs.replace("Object", "object")
        specs += f""""object": {{
                "layer_type": "Object",
                "layer_id": 1
            }},
            "state": ["{longest_path[-1]}"]
        """
        specs = '{' + specs + '}'
        return specs

    def generate(self):
        self.getTextEnvLayout()
        self.getTriplets()
        self.Canonicalization(self.env_ctx['environment_triplets'])
        env_specs = self.Triplets2Spec(self.env_ctx['environment_triplets'])
        return env_specs
    
    def verify(self, specs):
        try:
            print(specs)
            specs = specs.replace("'", '"')
            OpenSceneGraph(specs)
            print('[PASS]')
            return True, None
        except Exception as error:
            print('[Verify Error]', error)
            self.env_ctx['spec_error_message'] = error
            return False
    
    def loop(self):
        specs = self.generate()
        flag = self.verify(specs)
        trial = 0
        while not flag and trial < self.max_trial:
            feedback_trial = 0
            while not flag and feedback_trial < self.max_trial:
                specs = self.generate()
                flag = self.verify(specs)
                feedback_trial += 1
            trial += 1
            self.reset()
        return specs


autospec_instance = AutoSpec('hospital')
autospec_instance.loop()
# specs = autospec_instance.generate()
# print(specs)
# specs = """
# { "floor": {
#      "layer_type": "Place" ,
#      "layer_id": 3,
#      "contains": ["object"],
#      "connects to": ["store"]
# }, "store": {
#      "layer_type": "Place" ,
#      "layer_id": 3,
#      "contains": ["object"],
#      "connects to": ["store", "floor"]
# },"object": {
#                 "layer_type": "Object",
#                 "layer_id": 1
#             },
#             "state": ["store"]
#         }
# """
# OpenSceneGraph(specs)


