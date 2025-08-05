import os
import dgl
import torch
import json

import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 文件夹路径
bin_dir = '../../Data/MFCAD++/bin_global/'
labels_dir = '../../Data/MFCAD++/labels/'

# 获取所有的文件名
file_names = os.listdir(bin_dir)


for file_name in tqdm(file_names, desc="Processing MFCAD++"):
    if file_name.endswith('.bin_global'):
        # 对应的标签文件
        label_file_name = file_name[:-4] + '.json'

        # 构建完整的文件路径
        bin_path = os.path.join(bin_dir, file_name)
        label_path = os.path.join(labels_dir, label_file_name)

        # 读取JSON文件
        with open(label_path, 'r') as f:
            labels = json.load(f)

        # 将列表转换为张量
        node_labels = torch.tensor(labels)

        # 加载图
        g = dgl.load_graphs(bin_path)[0][0]  # 加载图

        # 将标签添加到图的节点数据中
        g.ndata['l'] = node_labels

        # 检查是否存在'f'键
        # if 'f' in g.ndata:
        #     del g.ndata['f']  # 删除'f'属性

        # 保存修改后的图
        dgl.save_graphs(bin_path, g)

print("All files processed.")
