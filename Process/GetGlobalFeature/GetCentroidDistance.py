import dgl
import torch
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


# 加载图数据
file_path = '../../Data/CADSynth/output/00000005.bin_global'  # 请确保这个路径是正确的
g = dgl.load_graphs(file_path)[0][0]

node_centroid_matrix = g.ndata["c"].type(torch.float32)


# 计算节点之间的距离矩阵
def compute_distance_matrix(node_centroid_matrix):
    # 获取节点数量
    num_nodes = node_centroid_matrix.shape[0]

    # 使用 PyTorch 计算两两节点之间的欧氏距离
    # 首先，将节点质心矩阵复制成形状为 [num_nodes, num_nodes, 3]
    expanded_a = node_centroid_matrix.unsqueeze(1).expand(-1, num_nodes, -1)
    expanded_b = node_centroid_matrix.unsqueeze(0).expand(num_nodes, -1, -1)

    # 计算距离
    distances = torch.norm(expanded_a - expanded_b, dim=2)

    return distances


# 调用函数计算距离矩阵
distance_matrix = compute_distance_matrix(node_centroid_matrix)

# 打印距离矩阵
print("distance matrix.shape:", distance_matrix.shape)
print(distance_matrix)