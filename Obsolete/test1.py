import torch

# 假设的数据
n_graph = 2
n_edge = 4  # 每张图最多有4条有效边
n_head = 3  # 注意力头的数量

# 创建 edge_padding_mask
edge_padding_mask = torch.tensor([
    [False, True, False, True],  # 图0的有效边是0和2
    [True, False, True, False]   # 图1的有效边是1和3
])

# 创建 edge_feat_
edge_feat_ = torch.tensor([
    [1, 2, 3],  # 边0的特征
    [4, 5, 6],  # 边1的特征
    [7, 8, 9],  # 边2的特征
    [10, 11, 12]  # 边3的特征
],dtype=torch.float32)

# 初始化 edge_feature
edge_feature = torch.zeros([n_graph, n_edge + 1, n_head], dtype=torch.float32)

# 确定有效边的位置
pos = torch.where(edge_padding_mask == False)


# 填充 edge_feature
edge_feature[pos] = edge_feat_

# 打印结果
print("edge_feature:")
print(edge_feature)