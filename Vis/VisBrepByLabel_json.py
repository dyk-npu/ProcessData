import argparse
import numpy as np
import json  # 导入 json 库，代替 pickle
from occwl.viewer import Viewer
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid

from occwl.edge import Edge
from occwl.solid import Solid
from occwl.shell import Shell
from occwl.compound import Compound

import torch
import dgl
import sys

import shutup
shutup.please()

# ... (其他函数定义保持不变) ...

# --- 修改：从 pkl 读取改为从 json 读取 ---
def load_labels_from_json(label_filename):
    """
    从 JSON 文件加载面标签。
    JSON 文件应包含一个键为字符串形式面索引、值为标签的字典。
    """
    with open(label_filename, "r", encoding="utf-8") as f:
        face_labels_dict = json.load(f)
    # 将键转换为整数并排序，以确保标签顺序与面的索引一致
    sorted_items = sorted(face_labels_dict.items(), key=lambda x: int(x[0]))
    # 提取排序后的标签值
    labels = np.array([v for k, v in sorted_items], dtype=int)
    return labels

# --- 新增：根据label给面上色 (此函数无需改动) ---
def display_faces_with_labels(viewer, face_obj_list, labels):
    # 纯色版本
    label_to_color = {
        0: (1.0, 0.0, 0.0),   # 纯红
        1: (0.0, 1.0, 0.0),   # 纯绿
        2: (0.0, 0.0, 1.0),   # 纯蓝
        3: (1.0, 1.0, 0.0),   # 纯黄
        4: (1.0, 0.0, 1.0),   # 纯紫
        5: (0.0, 1.0, 1.0),   # 纯青
    }

    default_color = (0.7, 0.7, 0.7)
    for idx, face in enumerate(face_obj_list):
        lab = labels[idx]
        color = label_to_color.get(lab, default_color)
        viewer.display(face, color=color)

def load_single_compound_from_step(step_filename):
    return Compound.load_from_step(step_filename)

def build_graph(entity, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    graph = face_adjacency(entity)

    graph_face_feat = []
    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
    graph_face_feat = np.asarray(graph_face_feat)

    graph_edge_feat = []
    edge_obj_list = []
    for edge_idx in graph.edges:
        edge = graph.edges[edge_idx]["edge"]
        if not edge.has_curve():
            continue
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
        edge_obj_list.append(edge)
    graph_edge_feat = np.asarray(graph_edge_feat)

    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    dgl_graph.edge_obj_list = edge_obj_list
    face_obj_list = [graph.nodes[face_idx]["face"] for face_idx in graph.nodes]
    dgl_graph.face_obj_list = face_obj_list
    return dgl_graph

def display_all_edges(viewer, edge_obj_list, highlight_indices=None, highlight_color=(0,1,0), default_color=(0.2,0.2,0.2), fake_width=14):
    highlight_indices = highlight_indices or []
    for idx, edge in enumerate(edge_obj_list):
        if idx in highlight_indices:
            viewer.display(edge, color=highlight_color)
        else:
            viewer.display(edge, color=default_color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Visualize and color selected edges of BRep solid"
    )
    parser.add_argument("--solid", type=str, help="Solid STEP file")
    # --- 修改：更新帮助信息为 json 文件 ---
    parser.add_argument("--label", type=str, help="Label json file")
    args = parser.parse_args()

    # 假设您的文件路径如下，如果通过命令行传入，可以注释掉这两行
    args.solid = r'D:\\CAD数据集\\项目\\GFR_Dataset_Final\\GFR_02664.step'
    # --- 修改：将标签文件名从 .pkl 改为 .json ---
    args.label = r'D:\CAD数据集\项目\GFR_dataset_label_hybrid\GFR_02664.json' # 假设您的json文件名与pkl一致

    # --- 主要修改部分：更灵活的实体加载逻辑 (此部分无需改动) ---
    print(f"正在加载 STEP 文件: {args.solid}...")
    compound = load_single_compound_from_step(args.solid)
    if compound is None:
        print(f"错误：无法加载 STEP 文件 '{args.solid}'。文件可能已损坏或格式不受支持。")
        sys.exit(1)

    compound = compound.scale_to_unit_box()

    entity_to_process = None

    solids = list(compound.solids())
    if solids:
        print(f"成功找到 {len(solids)} 个 Solid 实体。将处理第一个。")
        entity_to_process = solids[0]
    else:
        print("未找到 Solid。正在尝试寻找 Shell...")
        shells = list(compound.shells())
        if shells:
            print(f"成功找到 {len(shells)} 个 Shell 实体。将处理第一个。")
            entity_to_process = shells[0]
        else:
            print("错误：在文件中既没有找到 Solid 也没有找到 Shell。无法构建面邻接图。")
            sys.exit(1)
    
    graph = build_graph(entity_to_process, 10, 10, 10)

    # --- 后续逻辑修改 ---

    # --- 修改：调用新的 json 加载函数 ---
    labels = load_labels_from_json(args.label)
    assert len(labels) == len(graph.face_obj_list), f"label数({len(labels)})和面数({len(graph.face_obj_list)})不匹配"

    v = Viewer(backend="pyqt5")

    display_faces_with_labels(
        v,
        graph.face_obj_list,
        labels
    )

    display_all_edges(
        v,
        graph.edge_obj_list,
        highlight_indices=[],
        highlight_color=(1,0,0),
        default_color=(0.3,0.3,0.3),
        fake_width=18
    )

    v.fit()
    v.show()