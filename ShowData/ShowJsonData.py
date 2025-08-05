import json

import torch

# 假设你的JSON文件名为example.json
filename = '../Data/MFInstSeg/attr/output/20221121_154647_4.json'

# 使用with语句来确保文件正确关闭
with open(filename, 'r', encoding='utf-8') as file:
    # 加载JSON数据
    data = json.load(file)

    # print(data)
    graph = data[1]


    adjacency_graph = torch.tensor(graph["graph"]["edges"])
    print("adjacency_graph.shape:",adjacency_graph.shape)

    graph_face_attr = torch.tensor(graph["graph_face_attr"])
    print("graph_face_attr.shape:",graph_face_attr.shape)
    # print("graph_face_attr:",graph_face_attr)

    graph_face_grid = torch.tensor(graph["graph_face_grid"]).permute(0,2,3,1)
    print("graph_face_grid.shape:",graph_face_grid.shape)
    print("graph_face_grid:",graph_face_grid[0])

    graph_edge_attr = torch.tensor(graph["graph_edge_attr"])
    print("graph_edge_attr.shape:",graph_edge_attr.shape)
    # print("graph_edge_attr:",graph_edge_attr)


    graph_edge_grid = torch.tensor(graph["graph_edge_grid"]).permute(0,2,1)
    print("graph_edge_grid.shape:",graph_edge_grid.shape)
    print("graph_edge_grid:",graph_edge_grid)
