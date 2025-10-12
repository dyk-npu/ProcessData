import os
import pathlib
import signal
import tempfile
import traceback  # 导入新依赖，用于获取详细错误信息
from itertools import repeat
from multiprocessing import Manager
from multiprocessing.pool import Pool

import dgl
import numpy as np
import torch
import pyvista as pv
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.StlAPI import StlAPI_Reader
from OCC.Core.TopoDS import TopoDS_Shape
from occwl.graph import face_adjacency
from occwl.shell import Shell
from occwl.uvgrid import ugrid
from tqdm import tqdm

# --- 1. 用户配置区 ---
INPUT_BASE_DIR = r"E:\CAD数据集\MCB\MCB_A"
OUTPUT_FOLDER_NAME = "bin"
CURV_U_SAMPLES = 10
SURF_U_SAMPLES = 10
SURF_V_SAMPLES = 10
NUM_PROCESSES = 12
SKIP_EXISTING = True
MAX_FILE_SIZE_KB = 150000

# --- 2. 核心功能函数 (已加固) ---

def load_and_convert_mesh(file_path):
    tmp_path = None
    try:
        mesh = pv.read(file_path)
        if mesh.n_points == 0: return None
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmpfile:
            tmp_path = tmpfile.name
        mesh.save(tmp_path, binary=True)
        reader = StlAPI_Reader()
        shape = TopoDS_Shape()
        if not reader.Read(shape, tmp_path): return None
        sewing = BRepBuilderAPI_Sewing(1e-6)
        sewing.Add(shape)
        sewing.Perform()
        sewn_shape = sewing.SewedShape()
        return Shell(sewn_shape) if sewn_shape and not sewn_shape.IsNull() else Shell(shape)
    finally:
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

def build_graph(solid, config):
    """【已加固】根据给定的实体构建DGL图，增加了多重防御性检查。"""
    graph = face_adjacency(solid)
    # 检查1: 确保从实体成功生成了图，并且图中有节点（面）
    if not graph or not graph.nodes:
        raise ValueError("Failed to create a valid face-adjacency graph from the solid.")
    
    graph_face_feat, graph_edge_feat = [], []
    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]
        points = ugrid(face, "point", config['surf_u'], config['surf_v'])
        normals = ugrid(face, "normal", config['surf_u'], config['surf_v'])
        vis = ugrid(face, "visibility_status", config['surf_u'], config['surf_v'])
        mask = np.logical_or(vis == 0, vis == 2)
        graph_face_feat.append(np.concatenate((points, normals, mask), axis=-1))
    
    # 检查2: 确保成功提取了至少一个面的特征
    if not graph_face_feat:
        raise ValueError("Could not extract any face features from the graph.")

    # 只有图中有边时才处理边特征
    if graph.edges:
        for edge_idx in graph.edges:
            edge = graph.edges[edge_idx]["edge"]
            if not edge.has_curve(): continue
            points = ugrid(edge, "point", config['curv_u'])
            tangents = ugrid(edge, "tangent", config['curv_u'])
            graph_edge_feat.append(np.concatenate((points, tangents), axis=-1))
        
        # 检查3: 如果有边，要确保至少提取了一个边的特征
        if not graph_edge_feat:
            raise ValueError("Graph has edges, but failed to extract any edge features.")

    src, dst = zip(*graph.edges) if graph.edges else ([], [])
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(np.asarray(graph_face_feat)).float()
    
    if graph_edge_feat:
        dgl_graph.edata["x"] = torch.from_numpy(np.asarray(graph_edge_feat)).float()
    return dgl_graph

def process_one_file(args_tuple):
    """【已升级】处理单个文件的worker函数，现在能记录详细的错误堆栈。"""
    file_path, output_dir, config, stats = args_tuple
    output_file = output_dir / (file_path.stem + ".bin")

    try:
        if config['skip_existing'] and output_file.exists():
            with stats['lock']: stats['skipped_exist'] += 1
            return
        file_size_kb = file_path.stat().st_size / 1024
        if file_size_kb > config['max_size']:
            with stats['lock']: stats['skipped_size'] += 1
            return
        solid = load_and_convert_mesh(file_path)
        if solid is None: raise ValueError("Failed to load or convert mesh.")
        graph = build_graph(solid, config)
        dgl.save_graphs(str(output_file), [graph])
        with stats['lock']: stats['processed'] += 1
    except Exception:
        # --- 这是关键的升级 ---
        # 捕获任何异常，并记录完整的错误堆栈信息
        with stats['lock']:
            stats['failed'] += 1
            failed_log = stats['failed_log']
            # 使用 traceback.format_exc() 获取完整的错误报告
            error_details = traceback.format_exc()
            failed_log.append(f"File: {file_path}\n--- Traceback ---\n{error_details}\n-----------------\n")
            stats['failed_log'] = failed_log

def initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def run_batch_processing():
    """主执行函数"""
    train_input_dir = pathlib.Path(INPUT_BASE_DIR) / 'train'
    bin_output_dir = pathlib.Path(INPUT_BASE_DIR) / OUTPUT_FOLDER_NAME / 'train'
    log_file_path = pathlib.Path(INPUT_BASE_DIR) / OUTPUT_FOLDER_NAME / "error_log.txt"

    if not train_input_dir.is_dir():
        print(f"错误：输入目录不存在: {train_input_dir}")
        return

    bin_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输入目录: {train_input_dir}")
    print(f"输出目录: {bin_output_dir}")
    print(f"详细错误日志将保存在: {log_file_path}")

    print("\n正在扫描文件...")
    files_to_process = list(train_input_dir.rglob("*.obj")) + list(train_input_dir.rglob("*.stl"))
    if not files_to_process:
        print("未找到任何 .obj 或 .stl 文件。")
        return
    print(f"找到 {len(files_to_process)} 个待处理文件。")

    manager = Manager()
    stats = manager.dict({
        'processed': 0, 'skipped_exist': 0, 'skipped_size': 0, 'failed': 0,
        'lock': manager.Lock(), 'failed_log': manager.list()
    })
    config = {
        'curv_u': CURV_U_SAMPLES, 'surf_u': SURF_U_SAMPLES, 'surf_v': SURF_V_SAMPLES,
        'skip_existing': SKIP_EXISTING, 'max_size': MAX_FILE_SIZE_KB
    }
    tasks = []
    for f_path in files_to_process:
        relative_path = f_path.relative_to(train_input_dir)
        output_subdir = bin_output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        tasks.append((f_path, output_subdir, config, stats))

    print(f"即将使用 {NUM_PROCESSES} 个进程进行处理...")
    pool = Pool(NUM_PROCESSES, initializer)
    try:
        list(tqdm(pool.imap_unordered(process_one_file, tasks), total=len(tasks), desc="Processing files"))
    except KeyboardInterrupt:
        print("\n用户中断了程序。正在终止...")
    finally:
        pool.terminate()
        pool.join()

    print("\n" + "="*60)
    print("🎉 处理完成!")
    print(f"✅ 成功处理: {stats['processed']} 个文件")
    print(f"⏭️  跳过 (已存在): {stats['skipped_exist']} 个文件")
    print(f"⏭️  跳过 (文件太大): {stats['skipped_size']} 个文件")
    print(f"❌ 处理失败: {stats['failed']} 个文件")

    # 将详细的失败日志写入文件
    if stats['failed'] > 0:
        print(f"\n检测到 {stats['failed']} 个处理失败的文件，详细信息已写入日志。")
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(f"总计失败文件数: {stats['failed']}\n\n")
            for entry in stats['failed_log']:
                f.write(entry)

if __name__ == "__main__":
    run_batch_processing()