import json

import numpy as np
import torch

# 打开并加载JSON文件
with open('../Data/CADSynth/label/00000001.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 打印JSON数据
print(type(data[0]))

print( len(data[0]) )

# all_face_attr = []
# all_edge_attr = []
# for one_sample in data:
#     fn, graph = one_sample
#     all_face_attr.extend(graph["graph_face_attr"])
#     all_edge_attr.extend(graph["graph_edge_attr"])
fn, graph = data[0]
print("fn:",fn)
print("graph:",graph.keys())


# gra = torch.tensor(graph["graph"])
graph_face_attr = torch.tensor(graph["graph_face_attr"])
graph_face_grid = torch.tensor(graph["graph_face_grid"])

graph_edge_attr = torch.tensor(graph["graph_edge_attr"])
graph_edge_grid = torch.tensor(graph["graph_edge_grid"])


# print("gra:",gra.shape)
print("graph_face_attr.shape",graph_face_attr.shape)
print("graph_face_attr.shape",graph_face_attr)
# print("graph_face_grid.shape",graph_face_grid.shape)

print("graph_edge_attr.shape",graph_edge_attr.shape)
# print("graph_edge_grid.shape",graph_edge_grid.shape)
