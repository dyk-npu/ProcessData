# import os
# import shutil
#
#
# def rename_and_copy_files(parent_dir, target_dir):
#     # 创建目标目录，如果不存在的话
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
#     # 遍历父目录下的所有子目录
#     for subdir_name in os.listdir(parent_dir):
#         subdir_path = os.path.join(parent_dir, subdir_name,'STEP')
#
#         # 检查是否为目录
#         if os.path.isdir(subdir_path):
#             # 获取子目录下的所有文件
#             files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
#
#             # 对子目录下的文件进行重命名，并复制到目标目录
#             for index, filename in enumerate(files):
#                 # 文件的扩展名
#                 ext = os.path.splitext(filename)[1]
#
#                 # 新的文件名
#                 new_filename = f"{subdir_name}_{index + 1}{ext}"
#
#                 # 文件的完整路径
#                 old_file_path = os.path.join(subdir_path, filename)
#                 new_file_path = os.path.join(subdir_path, new_filename)
#
#                 # 重命名文件
#                 # os.rename(old_file_path, new_file_path)
#
#                 # 复制文件到目标目录
#                 target_file_path = os.path.join(target_dir, new_filename)
#                 shutil.copy(new_file_path, target_file_path)
#
#
# # 使用方法
# parent_directory = 'C:\\Users\\20268\Desktop\FavWave\FabWave'  # 替换成你的父目录路径
# target_directory = 'C:\\Users\\20268\Desktop\FabWave\step'  # 替换成你要复制的目标目录路径
# rename_and_copy_files(parent_directory, target_directory)


import os
import json

def get_file_names(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            # 提取文件名部分，忽略扩展名和末尾的数字及下划线
            base_name = '_'.join(filename.split('.')[0].split('_')[:-1])
            file_names.append(base_name)
    return list(set(file_names))  # 去除重复项并返回列表

def create_label_dict(file_names):
    label_dict = {}
    for i, name in enumerate(sorted(file_names), start=1):
        label_dict[name] = i
    return label_dict

def save_to_json(label_dict, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(label_dict, f, ensure_ascii=False, indent=4)

# 指定你的文件夹路径
directory_path = 'C:\\Users\\20268\Desktop\FabWave\step'
output_json_file = 'C:\\Users\\20268\Desktop\FabWave\labels.json'  # 输出的JSON文件名

file_names = get_file_names(directory_path)
label_dict = create_label_dict(file_names)
save_to_json(label_dict, output_json_file)

print(f"Label dictionary has been saved to {output_json_file}")


# target_directory = 'C:\\Users\\20268\Desktop\FabWave\step'  # 替换成你之前复制的目标目录路径
# output_json_file = 'C:\\Users\\20268\Desktop\FabWave\labels.json'  # 输出的JSON文件名
