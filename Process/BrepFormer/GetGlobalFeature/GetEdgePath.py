import dgl
import numpy as np

# 加载图数据
file_path = '../../Data/CADSynth/output/10795.bin_global'  # 请确保这个路径是正确的
g = dgl.load_graphs(file_path)[0][0]

# 设置最大距离
max_distance = 10  # 你可以根据实际情况设置这个值

# 获取节点数量
num_nodes = g.num_nodes()

# 创建一个三维矩阵来存储边 ID
edge_path_matrix = np.full((num_nodes, num_nodes, max_distance), -1, dtype=int)

# BFS 寻找最短路径及其边 ID
for source in range(num_nodes):
    visited = {}
    queue = [(source, [source], [])]  # (current node, path of nodes, path of edges)

    while queue:
        current_node, path, edge_path = queue.pop(0)

        if current_node not in visited:
            visited[current_node] = True

            # 如果路径长度没有超过最大距离，更新矩阵
            if len(edge_path) < max_distance:
                edge_path_matrix[source, current_node, :len(edge_path)] = edge_path

            # 检查是否到达其他未访问过的节点
            neighbors = g.successors(current_node)
            for neighbor in neighbors:
                edge_id = g.edge_ids(current_node, neighbor).item()
                queue.append((neighbor.item(), path + [neighbor.item()], edge_path + [edge_id]))

# 输出结果矩阵
print(edge_path_matrix[1])