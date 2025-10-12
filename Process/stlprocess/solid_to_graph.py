import argparse
import pathlib
import signal
from itertools import repeat
from multiprocessing.pool import Pool

import dgl
import numpy as np
import torch
from occwl.graph import face_adjacency
# from occwl.io import load_step
from occwl.uvgrid import ugrid, uvgrid
from tqdm import tqdm
from occwl.compound import Compound

import shutup

shutup.please()

def load_single_compound_from_step(step_filename):
    """
    Load data from a STEP file as a single compound

    Args:
        step_filename (str): Path to STEP file

    Returns:
        List of occwl.Compound: a single compound containing all shapes in
                                the file
    """
    return Compound.load_from_step(step_filename)



def load_step(step_filename):
    """Load solids from a STEP file

    Args:
        step_filename (str): Path to STEP file

    Returns:
        List of occwl.Solid: a list of solid models from the file
    """
    compound = load_single_compound_from_step(step_filename)
    return list(compound.solids())

def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    """
    (此函数保持不变)
    从B-rep实体构建一个带有UV-grid特征的面邻接图。
    """
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    # Compute the UV-grids for faces
    graph_face_feat = []
    # 使用tqdm为面处理添加进度条，方便调试慢文件
    for face_idx in tqdm(graph.nodes, desc="Processing faces", leave=False):
        # Get the B-rep face
        face = graph.nodes[face_idx]["face"]
        # Compute UV-grids
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
    graph_face_feat = np.asarray(graph_face_feat)

    # Compute the U-grids for edges
    graph_edge_feat = []
    # 使用tqdm为边处理添加进度条
    for edge_idx in tqdm(graph.edges, desc="Processing edges", leave=False):
        # Get the B-rep edge
        edge = graph.edges[edge_idx]["edge"]
        # Ignore degenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        # Concatenate channel-wise to form edge feature tensor
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    # Convert face-adj graph to DGL format
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    return dgl_graph


def process_one_file(arguments):
    """
    (此函数已修改)
    处理单个STEP文件，并将其保存在正确的输出子目录中。
    """
    # fn 是输入文件的完整路径
    # args 包含了基础的输入和输出目录
    fn, args = arguments
    
    base_input_path = pathlib.Path(args.input)
    base_output_path = pathlib.Path(args.output)
    
    try:
        # 1. 计算输入文件相对于基础输入目录的路径
        # 例如: 'class_A/model_01.stp'
        relative_path = fn.relative_to(base_input_path)
        
        # 2. 构建保留子文件夹结构的输出路径
        # 例如: 'E:/.../bin/class_A/model_01.stp'
        output_file_path = base_output_path.joinpath(relative_path)
        
        # 3. 将文件扩展名更改为.bin
        # 例如: 'E:/.../bin/class_A/model_01.bin'
        output_file_path = output_file_path.with_suffix(".bin")
        
        # 4. 如果输出子目录不存在，则创建它
        # 例如: 创建 'E:/.../bin/class_A'
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # 5. 加载和处理文件
        solids = load_step(fn)
        if not solids:
            print(f"警告: 在文件 {fn} 中未找到实体, 跳过。")
            return
        
        solid = solids[0]  # 假设每个文件只包含一个实体
        graph = build_graph(
            solid, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples
        )
        
        # 6. 保存图形文件
        dgl.data.utils.save_graphs(str(output_file_path), [graph])

    except Exception as e:
        print(f"处理文件 {fn} 时发生错误: {e}")


def initializer():
    """(此函数保持不变) Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process(args):
    """
    (此函数已修改)
    查找所有文件并启动多进程池。
    """
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output) # 基础输出目录
    
    # 确保基础输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 使用 rglob 进行递归搜索，查找所有子目录中的 .stp 和 .step 文件
    print(f"正在从 '{input_path}' 及其子目录中搜索文件...")
    step_files = list(input_path.rglob("*.st*p"))
    
    if not step_files:
        print(f"错误: 在 '{input_path}' 及其子目录中未找到任何 *.stp 或 *.step 文件。")
        return
        
    print(f"找到 {len(step_files)} 个文件待处理。")
    
    pool = Pool(processes=args.num_processes, initializer=initializer)
    try:
        # 使用 imap 以便 tqdm 能够正确显示进度
        list(tqdm(pool.imap(process_one_file, zip(step_files, repeat(args))), total=len(step_files), desc="Overall Progress"))
    except KeyboardInterrupt:
        print("\n检测到中断信号，正在终止进程...")
        pool.terminate()
        pool.join()
    finally:
        pool.close()
        pool.join()
        
    print(f"处理完成。")


def main():
    """
    (此函数已修改)
    更新了命令行参数的帮助说明。
    """
    parser = argparse.ArgumentParser(
        "将B-rep实体模型递归转换为带有UV-grid特征的面邻接图"
    )
    parser.add_argument("--input", default=r"E:\CAD数据集\Traceparts dataset\Step_models" ,type=str, help="包含STEP文件的根输入文件夹 (将进行递归搜索)")
    parser.add_argument("--output",default=r"E:\CAD数据集\Traceparts dataset\bin" ,type=str, help="保存DGL图BIN文件的根输出文件夹 (将保留原始目录结构)")
    parser.add_argument(
        "--curv_u_samples", type=int, default=10, help="每条曲线上采样点数"
    )
    parser.add_argument(
        "--surf_u_samples",
        type=int,
        default=10,
        help="每个曲面沿u方向的采样点数",
    )
    parser.add_argument(
        "--surf_v_samples",
        type=int,
        default=10,
        help="每个曲面沿v方向的采样点数",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=12,
        help="用于并行处理的进程数",
    )
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()