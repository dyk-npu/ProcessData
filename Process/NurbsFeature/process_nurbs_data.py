# -*- coding: utf-8 -*-

"""
CAD数据预处理脚本 (已添加日志功能)

功能:
    本脚本用于将STEP格式的3D CAD模型进行预处理，提取其几何与拓扑特征，
    并将其保存为Python的pickle (.pkl) 文件。

更新日志:
    - 增加了对运行过程的日志记录功能，可将所有输出保存到指定的.txt文件。
    - 修复了 'Solid' has no attribute 'load_from_step' 的严重BUG。
    - 增加了详细的诊断打印功能。
    - 适配了所有STEP文件都在单个文件夹内的简单数据结构。

如何运行:
    python preprocess_cad_data.py --log_file my_run_log.txt
"""

import os
import json
import pickle
import argparse
import logging  # <-- 1. 导入日志模块
import numpy as np
from tqdm import tqdm

# 核心依赖 (保持不变)
from occwl.solid import Solid
from occwl.compound import Compound
from occwl.shell import Shell
from occwl.uvgrid import ugrid, uvgrid
from occwl.entity_mapper import EntityMapper
from OCC.Core.TopoDS import topods, TopoDS_Face
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomConvert import geomconvert


################################################################################
#                                                                              #
#                      <-- 2. 新增日志配置函数 -->                              #
#                                                                              #
################################################################################

def setup_logging(log_file_path):
    """配置日志记录，使其同时输出到控制台和文件。"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置最低日志级别

    # 清除旧的处理器，防止重复记录
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建统一的日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 创建文件处理器 (用于写入到.txt文件)
    if log_file_path:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 创建流处理器 (用于在屏幕上打印)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


################################################################################
#                                                                              #
#                      数据加载与核心处理函数 (逻辑不变)                         #
#                                                                              #
################################################################################

def load_single_compound_from_step(step_filename):
    """
    Load data from a STEP file as a single compound
    """
    return Compound.load_from_step(step_filename)

def load_step(step_filename):
    """Load solids from a STEP file
    """
    compound = load_single_compound_from_step(step_filename)
    return list(compound.solids())

def load_step_files_from_folder(data_dir, split_json_path):
    """从单个文件夹加载STEP文件路径，并根据JSON文件进行筛选。"""
    try:
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)
    except FileNotFoundError:
        # <-- 4. 替换 print 为 logging -->
        logging.error(f"错误: 数据集划分文件未找到于 {split_json_path}")
        return []

    all_entries = split_data.get('train', []) + split_data.get('validation', []) + split_data.get('test', [])
    if not all_entries:
        logging.warning("警告: JSON文件中没有找到任何文件条目。")
        return []
    
    uids_to_process = set(all_entries)

    if not os.path.isdir(data_dir):
        logging.error(f"错误: 输入目录 '{data_dir}' 不存在。")
        return []

    found_file_paths = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.step', '.stp')):
            file_uid = os.path.splitext(filename)[0]
            if file_uid in uids_to_process:
                full_path = os.path.join(data_dir, filename)
                found_file_paths.append(full_path)
    
    found_uids = set([os.path.splitext(os.path.basename(p))[0] for p in found_file_paths])
    missing_uids = uids_to_process - found_uids
    if missing_uids:
        logging.warning(f"警告: JSON中指定的 {len(missing_uids)} 个文件未在输入目录中找到。")

    return found_file_paths

# ... (所有其他核心函数，仅替换 tqdm.write) ...
def get_nurbs_data(face, max_ctrl_setting, max_uvlength_setting, model_uid):
    try:
        topo_face, face_nurbs = face.topods_shape(), bspline_surface_from_face(face.topods_shape())
        ctrlPts = array_from_Array2OfPnt(face_nurbs.Poles())
        if max(ctrlPts.shape) > max_ctrl_setting:
            # <-- 4. 替换 tqdm.write 为 logging -->
            logging.warning(f"[模型: {model_uid}] 诊断: 控制点数量 {max(ctrlPts.shape)} 超出限制 {max_ctrl_setting}。")
            return None
        size_u, size_v = ctrlPts.shape[0], ctrlPts.shape[1]
        weights = np.ones((size_u, size_v, 1))
        for u_idx in range(1, size_u + 1):
            for v_idx in range(1, size_v + 1):
                weights[u_idx - 1, v_idx - 1] = face_nurbs.Weight(u_idx, v_idx)
        pw = np.concatenate([ctrlPts, weights], axis=-1)
        pw_data = np.zeros((max_ctrl_setting, max_ctrl_setting, 4))
        pw_data[:size_u, :size_v, :] = pw
        u_knotvector_raw = array_from_Array1OfReal(face_nurbs.UKnotSequence())
        v_knotvector_raw = array_from_Array1OfReal(face_nurbs.VKnotSequence())
        if max(len(u_knotvector_raw), len(v_knotvector_raw)) > max_uvlength_setting:
            logging.warning(f"[模型: {model_uid}] 诊断: 节点向量长度 {max(len(u_knotvector_raw), len(v_knotvector_raw))} 超出限制 {max_uvlength_setting}。")
            return None
        u_range = u_knotvector_raw.max() - u_knotvector_raw.min()
        u_knotvector = (u_knotvector_raw - u_knotvector_raw.min()) / u_range if u_range > 1e-6 else np.zeros_like(u_knotvector_raw)
        v_range = v_knotvector_raw.max() - v_knotvector_raw.min()
        v_knotvector = (v_knotvector_raw - v_knotvector_raw.min()) / v_range if v_range > 1e-6 else np.zeros_like(v_knotvector_raw)
        u_kv, v_kv = np.zeros(max_uvlength_setting), np.zeros(max_uvlength_setting)
        u_kv[:len(u_knotvector)], v_kv[:len(v_knotvector)] = u_knotvector, v_knotvector
        return pw_data, u_kv, v_kv
    except Exception as e:
        logging.warning(f"[模型: {model_uid}] 诊断: get_nurbs_data内部发生未知错误: {e}")
        return None

def extract_primitive(solid, max_ctrl_setting, max_uvlength_setting, model_uid):
    face_dict_raw, edge_dict_raw, edgeFace_IncM_raw = face_edge_adj(solid)
    if not edgeFace_IncM_raw:
        logging.warning(f"[模型: {model_uid}] 诊断: 邻接矩阵为空(edgeFace_IncM_raw)，模型可能不是有效实体或拓扑不符合要求。")
        return None
    face_dict, face_map = update_mapping(face_dict_raw)
    edge_dict, edge_map = update_mapping(edge_dict_raw)
    edgeFace_IncM_update = {edge_map[k]: [face_map[fi] for fi in v] for k, v in edgeFace_IncM_raw.items() if k in edge_map and all(fi in face_map for fi in v)}
    if not edgeFace_IncM_update:
        logging.warning(f"[模型: {model_uid}] 诊断: 邻接矩阵更新后为空，映射可能存在问题。")
        return None
    edgeFace_IncM = np.array(list(edgeFace_IncM_update.values()))
    num_faces = len(face_dict)
    faceEdge_IncM = [np.where(edgeFace_IncM == i)[0] for i in range(num_faces)]
    graph_face_pw, graph_face_ukv, graph_face_vkv, graph_face_pnts = {}, {}, {}, {}
    for face_idx, face in face_dict.items():
        nurbs_data = get_nurbs_data(face, max_ctrl_setting, max_uvlength_setting, model_uid)
        if nurbs_data is None:
            # 这条信息在 get_nurbs_data 内部已经记录过了，这里可以省略，避免重复
            # logging.warning(f"[模型: {model_uid}] 诊断: 因面 {face_idx} 的NURBS数据提取失败，放弃整个模型。")
            return None
        pw, u_kv, v_kv = nurbs_data
        graph_face_pw[face_idx], graph_face_ukv[face_idx], graph_face_vkv[face_idx] = pw, u_kv, v_kv
        graph_face_pnts[face_idx] = uvgrid(face, method="point", num_u=32, num_v=32)
    face_ctrlPts, face_ukv, face_vkv, face_pnts = (np.stack(list(d.values())) for d in [graph_face_pw, graph_face_ukv, graph_face_vkv, graph_face_pnts])
    graph_edge_feat = {idx: ugrid(edge, method="point", num_u=32) for idx, edge in edge_dict.items()}
    edge_pnts = np.stack(list(graph_edge_feat.values()))
    edge_corner_pnts = np.array([[p[0], p[-1]] for p in edge_pnts])
    return [face_ctrlPts, face_ukv, face_vkv, face_pnts, edge_pnts, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM]

# ... (update_mapping, face_edge_adj, bspline_surface_from_face, etc. remain unchanged) ...
def update_mapping(data_dict):
    dict_new, mapping = {}, {}
    if not data_dict: return dict_new, mapping
    max_idx = max(data_dict.keys())
    skipped_indices = np.array(sorted(list(set(np.arange(max_idx + 1)) - set(data_dict.keys()))))
    for idx, value in data_dict.items():
        skips = (skipped_indices < idx).sum()
        idx_new = idx - skips
        dict_new[idx_new], mapping[idx] = value, idx_new
    return dict_new, mapping

def face_edge_adj(shape):
    assert isinstance(shape, (Shell, Solid, Compound))
    mapper = EntityMapper(shape)
    face_dict = {mapper.face_index(face): face for face in shape.faces()}
    edgeFace_IncM, edge_dict, seen_edges = {}, {}, set()
    for edge in shape.edges():
        if not edge.has_curve(): continue
        try:
            tshape = edge.topods_shape().TShape()
            edge_hash = id(tshape)
            if edge_hash in seen_edges: continue
            seen_edges.add(edge_hash)
        except Exception:
            edge_hash = hash(edge)
            if edge_hash in seen_edges: continue
            seen_edges.add(edge_hash)
        try:
            connected_faces = list(shape.faces_from_edge(edge))
        except RuntimeError: continue
        if len(connected_faces) == 2 and not edge.seam(connected_faces[0]) and not edge.seam(connected_faces[1]):
            left_face, right_face = edge.find_left_and_right_faces(connected_faces)
            if left_face is None or right_face is None: continue
            edge_idx = mapper.edge_index(edge)
            edge_dict[edge_idx] = edge
            edgeFace_IncM[edge_idx] = [mapper.face_index(left_face), mapper.face_index(right_face)]
    return face_dict, edge_dict, edgeFace_IncM

def bspline_surface_from_face(face):
    if not isinstance(face, TopoDS_Face): raise TypeError("face must be a TopoDS_Face")
    nurbs_face = topods.Face(BRepBuilderAPI_NurbsConvert(face).Shape())
    surface = BRep_Tool.Surface(nurbs_face)
    return geomconvert.SurfaceToBSplineSurface(surface)

def array_from_Array2OfPnt(array):
    values = []
    for i in range(array.LowerRow(), array.UpperRow() + 1):
        row = [array.Value(i, j).Coord() for j in range(array.LowerCol(), array.UpperCol() + 1)]
        values.append(row)
    return np.asarray(values)

def array_from_Array1OfReal(array):
    if array is None: return np.empty(0)
    return np.asarray([array.Value(i) for i in range(array.Lower(), array.Upper() + 1)])
################################################################################
#                                                                              #
#                      脚本的主执行逻辑                                         #
#                                                                              #
################################################################################

def main(args):
    """脚本主函数，负责编排整个预处理流程。"""
    
    # <-- 4. 配置日志 -->
    setup_logging(args.log_file)
    
    logging.info("--- CAD数据预处理开始 ---")
    logging.info(f"输入数据目录: {args.input_dir}")
    logging.info(f"输出数据目录: {args.output_dir}")
    logging.info(f"数据集划分文件: {args.split_json}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    step_file_paths = load_step_files_from_folder(args.input_dir, args.split_json)
    if not step_file_paths:
        logging.error("未能加载任何文件路径，程序退出。")
        return
    logging.info(f"成功加载文件列表，共找到 {len(step_file_paths)} 个匹配的模型待处理。")

    success_count = 0
    fail_count = 0
    for step_file_path in tqdm(step_file_paths, desc="正在处理模型"):
        filename = os.path.basename(step_file_path)
        model_uid = os.path.splitext(filename)[0]
        output_pkl_path = os.path.join(args.output_dir, f"{model_uid}.pkl")
        
        if not args.overwrite and os.path.exists(output_pkl_path):
            continue
            
        try:
            shapes = load_step(step_file_path)
            
            if not shapes or not isinstance(shapes[0], Solid):
                logging.warning(f"[模型: {model_uid}] 诊断: STEP文件为空或第一个几何体不是有效的Solid。")
                fail_count += 1
                continue

            solid = shapes[0]
            
            extracted_data = extract_primitive(
                solid,
                max_ctrl_setting=args.max_control_points,
                max_uvlength_setting=args.max_knot_length,
                model_uid=model_uid
            )
            
            if extracted_data is not None:
                with open(output_pkl_path, 'wb') as f:
                    pickle.dump(extracted_data, f)
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logging.error(f"[模型: {model_uid}] 诊断: 在主循环中捕获到严重错误: {e}")
            fail_count += 1

    logging.info("--- 数据预处理完成 ---")
    logging.info(f"成功处理: {success_count} 个模型")
    logging.info(f"失败/跳过: {fail_count} 个模型")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="STEP CAD模型数据预处理脚本")
    
    parser.add_argument('--input_dir', type=str, default="/data_hdd/dev01/dyk/dyk_data/data_1/step",
                        help='包含所有 .step 文件的文件夹路径。')
    parser.add_argument('--output_dir', type=str, default="/data_hdd/dev01/dyk/dyk_data/data_1/NurbsFeature",
                        help='用于保存处理后的 .pkl 文件的目录。')
    parser.add_argument('--split_json', type=str, default="/data_hdd/dev01/dyk/model/Assembly-raw/Utils/ProcessFile/train_val_test_split.json",
                        help='指向 train_val_test_split.json 文件的路径。')
    
    # <-- 3. 新增log文件参数 -->
    parser.add_argument('--log_file', type=str, default='/data_hdd/dev01/dyk/model/Assembly-raw/Log/processing.log',
                        help='用于保存日志信息的文件路径。默认为 "processing.log"。')

    parser.add_argument('--max_control_points', type=int, default=50,
                        help='NURBS曲面允许的最大控制点数量。')
    parser.add_argument('--max_knot_length', type=int, default=50,
                        help='NURBS曲面允许的最大节点向量长度。')
    parser.add_argument('--overwrite', action='store_true',
                        help='如果设置此项，将覆盖已存在的 .pkl 文件。')

    args = parser.parse_args()
    main(args)
