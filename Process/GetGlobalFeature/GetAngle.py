import math

import dgl
import torch


# 加载图数据
file_path = '../../Data/CADSynth/output/00000005.bin_global'  # 请确保这个路径是正确的
g = dgl.load_graphs(file_path)[0][0]

node_feature = g.ndata["x"].type(torch.float32) # [node,10,10,7](x,y,z,nx,ny,nz,b)

print(node_feature.shape)



## 假设 node_feature 已经加载并且是一个 shape 为 (num_nodes, 10, 10, 7) 的张量
num_nodes = node_feature.shape[0]  # 获取节点数量

# 初始化一个空列表来保存每个节点的平均法向量
mean_normals_per_node = []

for i in range(num_nodes):
    # 提取法向量部分
    normals = node_feature[i, :, :, 3:6]  # 提取第4到第6维，即法向量部分

    # 获取表示是否被隐藏的维度
    hidden_status = node_feature[i, :, :, 6]  # 提取最后一维，即隐藏状态

    # 创建一个布尔掩码，用于过滤出未被隐藏的点
    mask = (hidden_status == 0)

    # 使用掩码过滤并计算法向量的平均值
    if torch.any(mask):  # 确保至少有一个点未被隐藏
        # 将mask扩展到与normals相同的维度
        mask_expanded = mask.unsqueeze(-1).expand_as(normals)
        # 使用掩码过滤法向量
        filtered_normals = normals[mask_expanded].view(-1, 3)
        # 计算法向量的平均值
        mean_normal = torch.mean(filtered_normals, dim=0)
    else:
        # 如果所有点都被隐藏，可以设置一个默认值或者跳过该节点
        mean_normal = torch.zeros(3)  # 或者其他处理方式

    mean_normals_per_node.append(mean_normal)

# 转换为张量
mean_normals_per_node = torch.stack(mean_normals_per_node)

print("Mean normals for each node:", mean_normals_per_node)
print(mean_normals_per_node.shape)

# 假设 mean_normals_per_node 已经是形状为 (num_nodes, 3) 的张量
num_nodes = mean_normals_per_node.shape[0]  # 获取节点数量

# 初始化一个 [n, n] 的矩阵来保存每个面之间的夹角
angle_matrix = torch.zeros((num_nodes, num_nodes))

# 计算任意两个面之间的夹角
for i in range(num_nodes):
    for j in range(i, num_nodes):
        # 计算法向量的点积
        dot_product = torch.dot(mean_normals_per_node[i], mean_normals_per_node[j])

        # 计算法向量的模长
        norm_i = torch.norm(mean_normals_per_node[i])
        norm_j = torch.norm(mean_normals_per_node[j])

        # 计算夹角的余弦值
        cos_theta = dot_product / (norm_i * norm_j)

        # 夹角的范围应该在 [-1, 1] 内，否则可能是由于数值不稳定导致的
        cos_theta = max(min(cos_theta, 1.0), -1.0)

        # 计算夹角（弧度）
        angle_radians = math.acos(cos_theta)

        # 转换为度数
        angle_degrees = math.degrees(angle_radians)

        # 填充角度矩阵
        angle_matrix[i, j] = angle_degrees
        angle_matrix[j, i] = angle_degrees  # 由于矩阵是对称的

print("Angle matrix:", angle_matrix)
print(angle_matrix.shape)
