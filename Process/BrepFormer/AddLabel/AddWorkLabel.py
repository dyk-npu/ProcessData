import os
import dgl
import torch
import json

from tqdm import tqdm
# 文件夹路径
bin_dir = '../../Data/work/single/bin_global'  # 替换为你的 bin_global 文件夹路径
labels_dir = '../../Data/work/single/label'  # 替换为你的 label 文件夹路径



# 获取所有的文件名
file_names = os.listdir(bin_dir)

for file_name in tqdm(file_names, desc="Processing WorkData"):
    if file_name.endswith('.bin_global'):
        # 对应的标签文件
        label_file_name = file_name[:-4] + '.json'

        # 构建完整的文件路径
        bin_path = os.path.join(bin_dir, file_name)
        label_path = os.path.join(labels_dir, label_file_name)

        # 读取JSON文件
        with open(label_path, 'r', encoding='utf-8-sig') as f:
            labels_dict = json.load(f)

        # 加载图
        g = dgl.load_graphs(bin_path)[0][0]  # 加载图
        num_node = g.num_nodes()

        max_key = num_node - 1

        # 补全缺失的键
        for i in range(max_key + 1):
            if str(i) not in labels_dict:
                labels_dict[str(i)] = 3

        # 获取标签值
        values = list(labels_dict.values())

        # 将列表转换为张量
        node_labels = torch.tensor(values)

        # 加载图
        g = dgl.load_graphs(bin_path)[0][0]  # 加载图

        # 将标签添加到图的节点数据中
        g.ndata['l'] = node_labels

        # 保存修改后的图
        dgl.save_graphs(bin_path, g)

print("All files processed.")