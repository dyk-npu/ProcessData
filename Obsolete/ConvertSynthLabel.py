import os
import dgl
import torch
from tqdm import tqdm

# 映射表
mapping_table = {
    0: 24,
    1: 6,
    2: 5,
    3: 3,
    4: 2,
    5: 4,
    6: 8,
    7: 9,
    8: 10,
    9: 22,
    10: 20,
    11: 17,
    12: 14,
    13: 13,
    14: 15,
    15: 0,
    16: 7,
    17: 1,
    18: 21,
    19: 19,
    20: 18,
    21: 16,
    22: 11,
    23: 12,
    24: 23
}

# 文件夹路径
bin_dir = '../Data/CADSynth/bin'

# 获取所有的文件名
file_names = [fn for fn in os.listdir(bin_dir) if fn.endswith('.bin_global')]

# 使用tqdm创建进度条
for file_name in tqdm(file_names, desc="Processing files"):
    bin_path = os.path.join(bin_dir, file_name)

    # 加载图
    graph, _ = dgl.load_graphs(bin_path)

    g = graph[0]
    # 检查是否存在'f'键
    if 'f' in g.ndata:
        # 应用映射表
        mapped_labels = torch.tensor([mapping_table[label.item()] for label in g.ndata['f']])

        # 更新'f'属性
        g.ndata['f'] = mapped_labels

    # 保存修改后的图
    dgl.save_graphs(bin_path, g)

print("All files processed.")