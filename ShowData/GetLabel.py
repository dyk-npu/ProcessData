# import json
#
#
# def find_keys_with_zero_value(json_file_path):
#     # 打开并加载JSON文件
#     with open(json_file_path, 'r', encoding='utf-8-sig') as file:
#         data = json.load(file)
#
#     # 查找值为0的键
#     keys_with_zero_value = [key for key, value in data.items() if value == 0]
#
#     return keys_with_zero_value
#
#
# # JSON文件路径
# json_file_path = '../Data/work/Test/single_1_666.json'
#
# # 调用函数并获取结果
# keys_with_zero = find_keys_with_zero_value(json_file_path)
#
# # 使用逗号连接键名，并打印出来
# print("Keys with a value of 0:", ', '.join(keys_with_zero))


import dgl
import torch

# 指定.bin文件的路径
bin_file_path = "../Data/CADSynth/bin_global/00000001.bin"

# print("bin_file_path:",type(bin_file_path))d
# 加载图
graphfile = dgl.data.utils.load_graphs(bin_file_path)
# print(type(graphfile) )
# print("graphfile:",graphfile)

graphs, geo = dgl.data.utils.load_graphs(bin_file_path)

# print("graphs,geo",graphs,geo)
# 获取第一个图（如果有多个图存储在文件中）
graph = graphs[0]
# print("geo:", geo)



node_feature_l = graph.ndata['l']
print("node_feature_l.shape:", node_feature_l.shape)
print("node_feature_l:", node_feature_l)

# 使用 torch.where 找到所有值为 24 的元素的索引
indices = torch.where(node_feature_l == 24)[0]  # [0] 是因为 torch.where 返回的是一个元组

# 输出这些索引
print("Indices of elements with value 24:", indices)