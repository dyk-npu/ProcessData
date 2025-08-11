import os
import pickle
import torch
import dgl

def pyg_to_dgl(pyg_data):
    """
    将一个 PyTorch Geometric (PyG) 的 Data 对象转换为 DGL 图。
    (此函数保持不变)
    """
    if pyg_data.edge_index is None or pyg_data.edge_index.numel() == 0:
        g = dgl.graph(([], []), num_nodes=pyg_data.num_nodes)
    else:
        g = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]), 
                      num_nodes=pyg_data.num_nodes)

    for key, value in pyg_data:
        if key in ['edge_index', 'num_nodes'] or not torch.is_tensor(value):
            continue
        
        if value.shape[0] == pyg_data.num_nodes:
            g.ndata[key] = value
        elif hasattr(pyg_data, 'num_edges') and pyg_data.num_edges is not None and value.shape[0] == pyg_data.num_edges:
            g.edata[key] = value
    
    return g


def process_and_save_new_structure(pickle_path, output_dir):
    """
    加载 pickle 文件，并按照新的指定结构保存数据。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    print(f"正在从 {pickle_path} 加载数据...")
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"错误：输入文件未找到 at {pickle_path}")
        return
    except Exception as e:
        print(f"错误：加载 pickle 文件时发生未知错误: {e}")
        return

    print("数据加载成功。")

    graphs_list = data.get('graphs')
    graph_files_list = data.get('graph_files')
    files_list = data.get('files')

    if not all([graphs_list, graph_files_list, files_list]):
        print("错误：pickle 文件中缺少 'graphs', 'graph_files', 或 'files' 等必需的键。")
        return

    num_entries = len(files_list)
    print(f"共找到 {num_entries} 个数据条目。开始处理...")

    for i in range(num_entries):
        pyg_graphs_triplet = graphs_list[i]
        graph_files_for_entry = graph_files_list[i]
        original_filename = files_list[i]
        
        # --- 步骤 1: 准备要保存的第一个对象 (只含前两个图) ---
        pyg_g1 = pyg_graphs_triplet[0]
        pyg_g2 = pyg_graphs_triplet[1]
        dgl_graphs_to_save = [pyg_to_dgl(pyg_g1), pyg_to_dgl(pyg_g2)]

        # --- 步骤 2: 准备要保存的第二个对象 (包含所有元数据的字典) ---
        pyg_g3 = pyg_graphs_triplet[2] # 获取第三个图用于提取数据
        labels_dict_to_save = {}
        
        # a) 处理 edge_attr: reshape并存入
        if hasattr(pyg_g3, 'edge_attr') and hasattr(pyg_g3, 'num_nodes_graph1') and hasattr(pyg_g3, 'num_nodes_graph2'):
            reshaped_attr = pyg_g3.edge_attr.reshape(pyg_g3.num_nodes_graph1, pyg_g3.num_nodes_graph2)
            labels_dict_to_save['edge_attr_matrix'] = reshaped_attr
        
        # b) 处理 edge_index: 直接存入
        if hasattr(pyg_g3, 'edge_index'):
            labels_dict_to_save['edge_index'] = pyg_g3.edge_index

        # c) 处理 graph_files: 编码为Tensor后存入
        for idx, filename in enumerate(graph_files_for_entry):
            key = f"graph_file_{idx}"
            encoded_str = torch.tensor(list(filename.encode('utf-8')), dtype=torch.uint8)
            labels_dict_to_save[key] = encoded_str

        # --- 步骤 3: 保存 ---
        base_filename = os.path.splitext(original_filename)[0]
        output_path = os.path.join(output_dir, f"{base_filename}.bin")
        
        dgl.save_graphs(output_path, dgl_graphs_to_save, labels_dict_to_save)
        
        if (i + 1) % 100 == 0 or (i + 1) == num_entries:
            print(f"({i + 1}/{num_entries}) 已处理 '{original_filename}' -> 保存至 '{output_path}'")

    print("\n所有数据处理完毕！")

def example_verify_new_structure(bin_path):
    """加载一个新结构的文件并验证其内容。"""
    if not os.path.exists(bin_path):
        print(f"文件不存在: {bin_path}")
        return

    print(f"\n--- 验证新结构文件: {bin_path} ---")
    
    graphs, labels = dgl.load_graphs(bin_path)
    
    print("\n--- 第一个对象 (图列表) ---")
    print(f"加载的图数量: {len(graphs)}")
    print("图列表内容:", graphs)

    print("\n--- 第二个对象 (元数据字典) ---")
    print(f"字典包含的键: {labels.keys()}")
    
    print("\n'edge_attr_matrix' 的形状:")
    print(labels['edge_attr_matrix'].shape)

    print("\n'edge_index' 的形状:")
    print(labels['edge_index'].shape)
    
    # 还原并打印文件名
    restored_filenames = []
    i = 0
    while f"graph_file_{i}" in labels:
        decoded_str = bytes(labels[f"graph_file_{i}"].tolist()).decode('utf-8')
        restored_filenames.append(decoded_str)
        i += 1
    print("\n还原的 'graph_files':")
    print(restored_filenames)

if __name__ == '__main__':
    input_pickle_file = "D:/CAD数据集/j1.0.0\joint/j1.0.0_preprocessed/joint/test.pickle"
    output_directory = "D:/CAD数据集/j1.0.0/joint/j1.0.0_preprocessed/joint/test_bin"

    process_and_save_new_structure(input_pickle_file, output_directory)
    
    # 验证环节
    # if os.path.exists(output_directory):
    #     example_bin_file = os.path.join(output_directory, "joint_set_03832.bin")
    #     if os.path.exists(example_bin_file):
    #          example_verify_new_structure(example_bin_file)