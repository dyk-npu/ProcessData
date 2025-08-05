import os
import dgl
import torch
from tqdm import tqdm

def get_adjacency_matrix(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中的所有 .bin_global 文件
    bin_files = [f for f in os.listdir(input_folder) if f.endswith('.bin_global')]
    for file_name in tqdm(bin_files, desc="Processing files"):
        file_path = os.path.join(input_folder, file_name)

        # 加载图
        g = dgl.load_graphs(file_path)[0][0]


        # 获取邻接矩阵
        adj_matrix = g.adjacency_matrix(transpose=False).to_dense()

        # 将邻接矩阵中的非零元素设置为1，以表示存在连接
        adj_matrix = (adj_matrix > 0).float()

        # 转换为 torch.float16 类型
        adj_matrix = adj_matrix.to(torch.float16)

        graphs, geo = dgl.load_graphs(file_path)

        geo['adj_matrix'] = adj_matrix

        # 生成输出文件路径
        output_file_path = os.path.join(output_folder, file_name)


        # 保存到.bin文件的第二个对象
        dgl.save_graphs(output_file_path, graphs, geo)

if __name__ == '__main__':
    input_folder = '../../Data/MFTR/bin_global'
    output_folder = '../../Data/MFTR/bin_global'

    get_adjacency_matrix(input_folder, output_folder)