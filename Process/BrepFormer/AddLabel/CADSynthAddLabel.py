import os
import json
import torch
import dgl
from tqdm import tqdm

# 定义文件夹路径
bin_dir = '../../BrepMFR-main/dataset/CADSynth/bin_my'
labels_dir = '../../BrepMFR-main/dataset/CADSynth/label'

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

# 获取所有的文件名
file_names = os.listdir(bin_dir)

successful_count = 0  # 成功处理的文件计数器
failed_count = 0      # 失败处理的文件计数器
wor_file = []         # 记录失败处理的文件名

for file_name in tqdm(file_names, desc="Processing CADSynth"):
    if file_name.endswith('.bin_global'):
        # 对应的标签文件
        label_file_name = file_name[:-4] + '.json'

        # 构建完整的文件路径
        bin_path = os.path.join(bin_dir, file_name)
        label_path = os.path.join(labels_dir, label_file_name)

        try:
            # 读取JSON文件
            with open(label_path, 'r') as f:
                data = json.load(f)

            # 将列表转换为张量
            node_labels = torch.tensor(data['labels'])

            mapped_labels = torch.tensor([mapping_table[label.item()] for label in node_labels])

            # 加载图
            g = dgl.load_graphs(bin_path)[0][0]  # 加载图

            # 检查标签数量是否与节点数量匹配
            if len(mapped_labels) != g.number_of_nodes():
                print("label_file_name:", label_file_name)
                print(f"Warning: Number of labels ({len(mapped_labels)}) does not match number of nodes ({g.number_of_nodes()}).")
                failed_count += 1
                wor_file.append(label_file_name)
                os.remove(bin_path)  # 删除不匹配的.bin文件
            else:
                # 将标签添加到图的节点数据中
                g.ndata['l'] = mapped_labels

                # 保存修改后的图
                dgl.save_graphs(bin_path, g)
                successful_count += 1  # 增加成功处理的文件计数器
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            failed_count += 1  # 增加失败处理的文件计数器

print("Successful count:", successful_count)
print("Failed count:", failed_count)
print("wor_file:", wor_file)
print("All files processed.")