# import os
# import dgl
#
# def check_and_delete_inf_bin_files(directory):
#     # 遍历指定目录下的所有文件
#     for filename in os.listdir(directory):
#         if filename.endswith('.bin'):
#             file_path = os.path.join(directory, filename)
#             try:
#                 # 加载图和属性
#                 graphs, geo = dgl.data.utils.load_graphs(file_path)
#                 # 获取最短距离矩阵
#                 shortest_distance = geo.get('shortest_distance_matrix', None)
#                 if shortest_distance is not None:
#                     # 检查矩阵中是否有 inf
#                     if not (shortest_distance == float('inf')).any():
#                         print(f"No inf found in {filename}")
#                     else:
#                         print(f"Inf found in {filename}, deleting...")
#                         os.remove(file_path)
#                 else:
#                     print(f"No 'shortest_distance_matrix' found in {filename}")
#             except Exception as e:
#                 print(f"Failed to process {filename}: {e}")
#
# # 使用方法示例
# directory = '../../BrepMFR-main/dataset/MFInstSeg/bin_global'  # 替换为你的文件夹路径
# check_and_delete_inf_bin_files(directory)

# import os
# import shutil
#
# def rename_files(directory):
#     # 遍历目录中的所有文件
#     for filename in os.listdir(directory):
#         if filename.endswith('.bin'):
#             # 获取文件的完整路径
#             file_path = os.path.join(directory, filename)
#             # 获取文件名（不包括路径）
#             base_name, _ = os.path.splitext(filename)
#             # 去掉文件名中的 _result
#             new_name = base_name.replace('_result', '') + '.bin'
#             # 获取新的文件路径
#             new_file_path = os.path.join(directory, new_name)
#             # 重命名文件
#             shutil.move(file_path, new_file_path)
#             print(f'Renamed {filename} to {new_name}')
#
# # 指定目录
# directory = "../../BrepMFR-main/dataset/MFTRCAD/bin_global"
#
# # 调用函数
# rename_files(directory)

import os
import dgl
import torch

# 确保 device 是正确的设备（CPU 或 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def has_l_key(file_path):
    try:
        gs , g2 = dgl.data.utils.load_graphs(file_path)
        # 检查 'l' 键是否存在于节点数据中
        g = gs[0]
        return "l" in g.ndata
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False

def check_and_remove_invalid_files(directory):
    invalid_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                if not has_l_key(file_path):
                    print(f"Removing file: {file_path}")
                    invalid_files.append(file)
                    os.remove(file_path)

    if invalid_files:
        print("The following files were removed due to missing 'l' parameter:")
        for file in invalid_files:
            print(file)
    else:
        print("All files have the 'l' parameter.")

if __name__ == "__main__":
    bin_directory = '../../BrepMFR-main/dataset/MFTRCAD/bin_global'  # 替换为你的 bin_global 文件夹路径
    check_and_remove_invalid_files(bin_directory)



