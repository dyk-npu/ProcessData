import os
import dgl
import torch
import glob

def load_my_data(bin_path):
    """
    辅助函数：加载我们之前创建的.bin文件。
    它会隐藏所有解码细节，直接返回图列表和包含元数据的字典。
    """
    if not os.path.exists(bin_path): return None, None
    
    graphs, labels_with_tensors = dgl.load_graphs(bin_path)
    
    restored_filenames = []
    i = 0
    while f"graph_file_{i}" in labels_with_tensors:
        decoded_str = bytes(labels_with_tensors.pop(f"graph_file_{i}").tolist()).decode('utf-8')
        restored_filenames.append(decoded_str)
        i += 1
        
    final_labels = labels_with_tensors
    final_labels['graph_files'] = restored_filenames
            
    return graphs, final_labels

def process_and_rebuild_bins(joint_files_dir, individual_graph_dir, output_dir, error_log_file="error_log.txt"):
    """
    执行最终的工作流：
    1. 替换图。
    2. 填充 edge_attr_matrix。
    3. 重建 edge_index 以匹配新维度。
    """
    # --- 初始化 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
        
    failed_files = []
    joint_files = glob.glob(os.path.join(joint_files_dir, '*.bin'))
    if not joint_files:
        print(f"错误：在目录 '{joint_files_dir}' 中没有找到任何 .bin 文件。")
        return

    num_files = len(joint_files)
    print(f"找到 {num_files} 个联合.bin文件，开始处理...")

    # --- 主循环，处理每个联合文件 ---
    for i, joint_bin_path in enumerate(joint_files):
        try:
            # (1) 加载联合文件并获取替换图
            _, metadata = load_my_data(joint_bin_path)
            if metadata is None or 'graph_files' not in metadata or len(metadata['graph_files']) != 2:
                raise ValueError("元数据格式不正确或缺少'graph_files'。")
            
            graph_filenames = metadata['graph_files']
            
            new_graph_paths = [os.path.join(individual_graph_dir, f"{os.path.splitext(fname)[0]}.bin") for fname in graph_filenames]
            
            loaded_individual_graphs = []
            for path in new_graph_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"对应的单个图文件未找到: {path}")
                individual_graph_list, _ = dgl.load_graphs(path)
                loaded_individual_graphs.append(individual_graph_list[0])

            # (2) 获取替换后的图并计算新维度
            new_graphs_list = loaded_individual_graphs
            graph1, graph2 = new_graphs_list[0], new_graphs_list[1]
            sum1 = graph1.num_nodes() + graph1.num_edges()
            sum2 = graph2.num_nodes() + graph2.num_edges()
            
            # (3a) **填充 edge_attr_matrix** (逻辑不变)
            original_matrix = metadata['edge_attr_matrix']
            orig_rows, orig_cols = original_matrix.shape
            padded_matrix = torch.zeros((sum1, sum2), dtype=original_matrix.dtype, device=original_matrix.device)
            padded_matrix[0:orig_rows, 0:orig_cols] = original_matrix
            metadata['edge_attr_matrix'] = padded_matrix

            # (3b) **核心修改：重建 edge_index**
            # 我们不再关心旧的edge_index，而是根据sum1和sum2从头创建一个新的、完整的连接。
            # 这会创建一个包含所有 (sum1, sum2) 组合的完整二部图的边索引。
            new_edge_index = torch.cartesian_prod(
                torch.arange(sum1),
                torch.arange(sum2)
            ).t() # 使用 .t() 转置以获得 [2, sum1 * sum2] 的形状

            # (3c) 更新元数据字典中的 edge_index
            metadata['edge_index'] = new_edge_index
            
            # --- 最后一步: 重新保存 ---
            final_labels_to_save = {}
            for key, value in metadata.items():
                if torch.is_tensor(value):
                    final_labels_to_save[key] = value
            for idx, filename_str in enumerate(metadata['graph_files']):
                final_labels_to_save[f"graph_file_{idx}"] = torch.tensor(list(filename_str.encode('utf-8')), dtype=torch.uint8)

            output_path = os.path.join(output_dir, os.path.basename(joint_bin_path))
            dgl.save_graphs(output_path, new_graphs_list, final_labels_to_save)

        except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
            print(f"处理文件 '{os.path.basename(joint_bin_path)}' 时发生错误，已跳过。错误: {e}")
            failed_files.append(joint_bin_path)
            continue
            
        if (i + 1) % 100 == 0 or (i + 1) == num_files:
            print(f"({i + 1}/{num_files}) 已处理 '{os.path.basename(joint_bin_path)}'")
            
    # --- 结尾：写入错误日志 ---
    if failed_files:
        print(f"\n处理完成，但有 {len(failed_files)} 个文件处理失败。详情请查看 '{error_log_file}'。")
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write("以下文件在处理过程中失败：\n")
            for path in failed_files:
                f.write(f"{path}\n")
    else:
        print("\n所有文件处理成功！")

if __name__ == '__main__':
    # --- 配置 ---
    JOINT_FILES_INPUT_DIR = '/workspace/data/JointData/Preprocessed/bin/val' 
    INDIVIDUAL_GRAPHS_DIR = '/workspace/data/JointData/bin' 
    FINAL_OUTPUT_DIR = '/workspace/data/JointData/Preprocessed/bin/val_padding'


    # --- 执行核心工作流 ---
    process_and_rebuild_bins(
        joint_files_dir=JOINT_FILES_INPUT_DIR,
        individual_graph_dir=INDIVIDUAL_GRAPHS_DIR,
        output_dir=FINAL_OUTPUT_DIR
    )
