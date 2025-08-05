import dgl
import torch
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


# 加载图数据
file_path = '../../Data/CADSynth/output/10795.bin_global'  # 请确保这个路径是正确的
g = dgl.load_graphs(file_path)[0][0]

# 将DGL图转换为NetworkX图
G = g.to_networkx()

# 使用 all_pairs_shortest_path_length 获取所有节点间的最短路径长度
lengths = dict(nx.all_pairs_shortest_path_length(G))

# 创建一个 n x n 的距离矩阵
n = G.number_of_nodes()
distance_matrix = [[lengths[i].get(j, float('inf')) for j in range(n)] for i in range(n)]

max_distance = -np.inf
# 遍历距离矩阵
for i in range(n):
    for j in range(n):
        if distance_matrix[i][j] > max_distance:
            max_distance = distance_matrix[i][j]

print("max distance:", max_distance)
# 打印距离矩阵
print("max_distance.shape:",torch.tensor(distance_matrix).shape)

for row in distance_matrix:
    print(row)