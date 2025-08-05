# import os
# import shutil
#
#
# #分开
# # 设置源文件夹路径
# folder_path = '../../dataset/WorkData/single'  # 替换为你的文件夹路径
#
# # 创建目标文件夹
# label_folder = os.path.join(folder_path, 'label')
# step_folder = os.path.join(folder_path, 'step')
#
# # 确保目标文件夹存在
# os.makedirs(label_folder, exist_ok=True)
# os.makedirs(step_folder, exist_ok=True)
#
# # 遍历源文件夹中的所有文件
# for file_name in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, file_name)
#
#     # 检查是否是文件
#     if os.path.isfile(file_path):
#         # 根据文件扩展名移动文件
#         if file_name.endswith('.json'):
#             shutil.move(file_path, os.path.join(label_folder, file_name))
#         elif file_name.endswith('.step'):
#             shutil.move(file_path, os.path.join(step_folder, file_name))
#
# print("文件已成功移动到对应的文件夹中。")



# 重命名1
# import os
#
# # 设置 label 文件夹路径
# label_folder = '../../Data/work/single/label'  # 替换为你的 label 文件夹路径
#
# # 遍历 label 文件夹中的所有 json 文件
# for filename in os.listdir(label_folder):
#     if filename.endswith('.json') and filename.startswith('面编号映射'):
#         old_filepath = os.path.join(label_folder, filename)
#
#         # 提取文件名中的数字部分
#         number_part = filename[len('面编号映射'):]
#         new_filename = f'single{number_part}'
#
#         new_filepath = os.path.join(label_folder, new_filename)
#
#         # 如果新的文件名不存在，则进行重命名
#         if not os.path.exists(new_filepath):
#             os.rename(old_filepath, new_filepath)
#         else:
#             print(f"{new_filepath} 已经存在，跳过重命名")
#
# print("文件重命名已完成。")


#重命名2

# import os
#
# # 设置 label 文件夹路径
# label_folder = '../../Data/work/single/step'  # 替换为你的 label 文件夹路径
#
# # 遍历 label 文件夹中的所有 json 文件
# for filename in os.listdir(label_folder):
#     if filename.endswith('.step') and filename.startswith('平移'):
#         old_filepath = os.path.join(label_folder, filename)
#
#         # 提取文件名中的数字部分
#         number_part = filename[len('平移'):]
#         new_filename = f'single{number_part}'
#
#         new_filepath = os.path.join(label_folder, new_filename)
#
#         # 如果新的文件名不存在，则进行重命名
#         if not os.path.exists(new_filepath):
#             os.rename(old_filepath, new_filepath)
#         else:
#             print(f"{new_filepath} 已经存在，跳过重命名")
#
# print("文件重命名已完成。")



# label减一

import json
import os

# 设置 label 文件夹路径
label_folder = '../../Data/work/single/label'  # 替换为你的 label 文件夹路径

# 遍历 label 文件夹中的所有 json 文件
for filename in os.listdir(label_folder):
    if filename.endswith('.json'):
        filepath = os.path.join(label_folder, filename)
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        # 修改标签编号
        for key in list(data.keys()):
            new_key = int(key) - 1
            value = data.pop(key)
            data[str(new_key)] = value

        # 重新写入文件
        with open(filepath, 'w', encoding='utf-8-sig') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

print("文件内容修改已完成。")