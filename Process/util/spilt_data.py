import os
import random

# 设置文件夹路径
folder_path = '../dataset/WorkData/double/bin_global'  # 替换为你的文件夹路径

# 获取所有 .bin_global 文件
files = [f for f in os.listdir(folder_path) if f.endswith('.bin_global')]

# 随机打乱文件列表
random.shuffle(files)

# 定义划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 计算每个集合的大小
total_files = len(files)
train_size = int(total_files * train_ratio)
val_size = int(total_files * val_ratio)

# 划分数据集
train_files = files[:train_size]
val_files = files[train_size:train_size + val_size]
test_files = files[train_size + val_size:]

# 定义输出文件名
output_train = '../dataset/WorkData/double/train.txt'
output_val = '../dataset/WorkData/double/val.txt'
output_test = '../dataset/WorkData/double/test.txt'

# 写入训练集文件名（不包括 .bin_global 后缀）
with open(output_train, 'w') as f:
    for file_name in train_files:
        base_name = os.path.splitext(file_name)[0]  # 去掉 .bin_global 后缀
        f.write(base_name + '\n')

# 写入验证集文件名（不包括 .bin_global 后缀）
with open(output_val, 'w') as f:
    for file_name in val_files:
        base_name = os.path.splitext(file_name)[0]  # 去掉 .bin_global 后缀
        f.write(base_name + '\n')

# 写入测试集文件名（不包括 .bin_global 后缀）
with open(output_test, 'w') as f:
    for file_name in test_files:
        base_name = os.path.splitext(file_name)[0]  # 去掉 .bin_global 后缀
        f.write(base_name + '\n')

print(f"训练集文件已保存至 {output_train}")
print(f"验证集文件已保存至 {output_val}")
print(f"测试集文件已保存至 {output_test}")