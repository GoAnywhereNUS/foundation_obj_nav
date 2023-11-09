import os
import networkx as nx
# import openai
# import dotenv
# import pickle
# import retry
# import numpy as np
# dotenv.load_dotenv(".env", override=True)
import  pdb 

# G = nx.Graph()
# G.add_node(0, label = 'kitchen', node_type = 'room')
# G.add_node(1, label = 'sink', node_type = 'object')
# G.add_edge(0,1)
# pdb.set_trace()

class SceneGraph:
    def __init__(self):
        self.G = nx.Graph()

    def add_node_edge(self, idx: int, node_label: str, node_node_type: str, linked_node_idx = None):
        if not (node_node_type in ['room', 'object', 'entrance']):
            print(f'[Scene Graph Error]: Wrong {node_node_type}')
            return

        self.G.add_node(idx, label = node_label, node_type = node_node_type)
        if linked_node_idx != None:
            if linked_node_idx in self.G.nodes:
                self.G.add_edge(idx, linked_node_idx)
            else:
                print(f'[Scene Graph Error]: No node {linked_node_idx}')

    def graph2text(self):
        graph_describe = 'The following is the layout of Apartment. There are '
        connect_describe = ''
        for node_id in self.G.nodes:
            if self.G.nodes[node_id]['node_type'] == 'room':
                graph_describe += self.G.nodes[node_id]['label']
                graph_describe += ", "
            elif self.G.nodes[node_id]['node_type'] == 'entrance':
                connected_nodes = list(self.G.neighbors(node_id))
                room1_id, room2_id = connected_nodes[0], connected_nodes[1]
                text = self.G.nodes[room1_id]['label'] + ' is connected to ' + self.G.nodes[room2_id]['label'] + '. '
                connect_describe += text
        graph_describe = graph_describe[:-2] + '. '
        graph_describe += connect_describe
        return graph_describe

    def query_llm(self, current_idx):
        print(self.G.nodes)
        print(self.G.edges)


sg = SceneGraph()
sg.add_node_edge(0,'kitchen', 'room')
sg.add_node_edge(1, 'sink', 'object', 0)
sg.add_node_edge(-1, 'door', 'entrance', 0)
sg.add_node_edge(3, 'dining room', 'room', -1)

sg.query_llm(0)
txt = sg.graph2text()
print(txt)
# pdb.set_trace()


# def query_llm(method: int, object_clusters: list, goal: str, save_reasoning: bool = False, reasoning_file: str = "", timestep: int = 0, reasoning_enabled: bool = True) -> list:
#     """
#     Query the LLM fore a score and a selected goal. Returns a list of language scores for each target point
#     method = 0 uses the naive single sample LLM and binary scores of 0 or 1
#     method = 1 uses the sampling based approach and gives scores between 0 and 1
#     """

#     # Convert object clusters to a tuple of tuples so we can hash it and get unique elements
#     object_clusters_tuple = [tuple(x) for x in object_clusters]
#     # Remove empty clusters and duplicate clusters
#     query = list(set(tuple(object_clusters_tuple)) - set({tuple([])}))

#     if method == 0:
#             try:
#                 goal_id, reasoning = ask_gpt(goal, query)
#             except Exception as excptn:
#                 goal_id, reasoning = 0, "GPT failed"
#                 print("GPT failed:", excptn)
#             if goal_id != 0:
#                 goal_id = np.argmax([1 if x == query[goal_id - 1] else 0 for x in object_clusters_tuple]) + 1
#             language_scores = [0] * (len(object_clusters_tuple) + 1)
#             language_scores[goal_id] = 1
#     elif method == 1:
#         try: 
#             answer_counts, reasoning = ask_gpts(goal, query)
#         except Exception as excptn:
#             answer_counts, reasoning = {}, "GPTs failed"
#             print("GPTs failed:", excptn)
#         language_scores = [0] * (len(object_clusters_tuple) + 1)
#         for key, value in answer_counts.items():
#             if key != 0:
#                 for i, x in enumerate(object_clusters_tuple):
#                     if x == query[key - 1]:
#                         language_scores[i + 1] = value
#             else:
#                 language_scores[0] = value
#     elif method == 2:
#         try:
#             answer_counts, reasoning = ask_gpts_v2(goal, query, positives=True, reasoning_enabled=reasoning_enabled)
#         except Exception as excptn:
#             answer_counts, reasoning = {}, "GPTs failed"
#             print("GPTs failed:", excptn)
#         language_scores = [0] * len(object_clusters_tuple)
#         for key, value in answer_counts.items():
#             for i, x in enumerate(object_clusters_tuple):
#                 if x == query[key - 1]:
#                     language_scores[i] = value

#                 # Save reasoning to a file
#         if save_reasoning:
#             with open(reasoning_file, "a") as f:
#                 f.write(f"Timestep: {timestep}\n")
#                 f.write(f"Goal: {goal}\n")
#                 f.write(f"Query: {query}\n")
#                 f.write(f"Answer counts: {answer_counts}\n")
#                 f.write(f"Reasoning: {reasoning}\n")
#                 f.write(f"Object clusters: {object_clusters}\n")
#                 f.write(f"Language scores: {language_scores}\n\n")
#     elif method == 3:
#         try:
#             answer_counts, reasoning = ask_gpts_v2(goal, query, positives=False)
#         except Exception as excptn:
#             answer_counts, reasoning = {}, "GPTs failed"
#             print("GPTs failed:", excptn)
#         language_scores = [0] * len(object_clusters_tuple)
#         for key, value in answer_counts.items():
#             for i, x in enumerate(object_clusters_tuple):
#                 if x == query[key - 1]:
#                     language_scores[i] = value
            
#         # Save reasoning to a file
#         if save_reasoning:
#             with open(reasoning_file, "a") as f:
#                 f.write(f"Timestep: {timestep}\n")
#                 f.write(f"Goal: {goal}\n")
#                 f.write(f"Query: {query}\n")
#                 f.write(f"Answer counts: {answer_counts}\n")
#                 f.write(f"Reasoning: {reasoning}\n")
#                 f.write(f"Object clusters: {object_clusters}\n")
#                 f.write(f"Language scores: {language_scores}\n\n")
#     else:
#         raise Exception("Invalid method")
    
#     # The first element of language scores is the scores for uncertain, the last n-1 correspond to the semantic scores for each point
#     return language_scores, reasoning


















