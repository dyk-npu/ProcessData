import argparse
import numpy as np
import pickle
from occwl.viewer import Viewer
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid

from occwl.edge import Edge
from occwl.solid import Solid
from occwl.compound import Compound

import torch
import dgl

import shutup
shutup.please()

# --- 新增：读取标签
def load_labels_from_pkl(label_filename):
    with open(label_filename, "rb") as f:
        data = pickle.load(f)  # numpy.ndarray，里面元素是dict
    face_labels_dict = data['face_labels']
    sorted_items = sorted(face_labels_dict.items(), key=lambda x: int(x[0]))
    labels = np.array([v for k,v in sorted_items], dtype=int)
    return labels

# --- 新增：根据label给面上色，替代display_all_faces
def display_faces_with_labels(viewer, face_obj_list, labels):
    # 这里定义固定的颜色映射（RGB），颜色可以根据需要改淡一点
    label_to_color = {
        0: (0.8, 0.4, 0.4),   # 淡红
        1: (0.4, 0.8, 0.4),   # 淡绿
        2: (0.4, 0.4, 0.8),   # 淡蓝
        3: (0.8, 0.8, 0.4),   # 淡黄
        4: (0.8, 0.4, 0.8),   # 淡紫
        5: (0.4, 0.8, 0.8),   # 淡青
        # 你可以继续定义更多颜色...
    }
    # 如果遇到未定义label，则给它灰色
    default_color = (0.7, 0.7, 0.7)

    for idx, face in enumerate(face_obj_list):
        lab = labels[idx]
        color = label_to_color.get(lab, default_color)
        viewer.display(face, color=color)

def load_single_compound_from_step(step_filename):
    return Compound.load_from_step(step_filename)

def load_step(step_filename):
    compound = load_single_compound_from_step(step_filename)
    return list(compound.solids())

def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    graph = face_adjacency(solid)

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
    # 保存面对象
    face_obj_list = [graph.nodes[face_idx]["face"] for face_idx in graph.nodes]
    dgl_graph.face_obj_list = face_obj_list
    return dgl_graph

def display_all_faces(viewer, face_obj_list, highlight_indices=None, highlight_color=(1,0,0), default_color=(0.7,0.7,0.7)):
    highlight_indices = highlight_indices or []
    for idx, face in enumerate(face_obj_list):
        if idx in highlight_indices:
            viewer.display(face, color=highlight_color)
        else:
            viewer.display(face, color=default_color)

def display_edge_with_fake_width(viewer, edge, color=(1,0,0), width=12, n_sample=80):
    points = ugrid(edge, method="point", num_u=n_sample)
    viewer.display_points(points, color=color, marker="point", scale=width)

def display_all_edges(viewer, edge_obj_list, highlight_indices=None, highlight_color=(0,1,0), default_color=(0.2,0.2,0.2), fake_width=14):
    highlight_indices = highlight_indices or []
    for idx, edge in enumerate(edge_obj_list):
        if idx in highlight_indices:
            viewer.display(edge, color=highlight_color)
            display_edge_with_fake_width(viewer, edge, color=highlight_color, width=fake_width, n_sample=100)
        else:
            viewer.display(edge, color=default_color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Visualize and color selected edges of BRep solid"
    )
    parser.add_argument("--solid", type=str, help="Solid STEP file")
    parser.add_argument("--label", type=str, help="Label pkl file")  # 新增label参数
    args = parser.parse_args()

    # 如果你要传命令行参数，就注释掉下面这行
    args.solid = 'D:/CAD数据集/项目/GFR_Dataset/GFR_00028.step'
    args.label = 'D:/CAD数据集/项目/GFR_TrainingData_Modify/GFR_00028.pkl'

    solid = load_step(args.solid)[0]
    solid = solid.scale_to_unit_box()
    graph = build_graph(solid, 10, 10, 10)

    # 新增读取标签
    labels = load_labels_from_pkl(args.label)
    assert len(labels) == len(graph.face_obj_list), f"label数({len(labels)})和面数({len(graph.face_obj_list)})不匹配"

    v = Viewer(backend="pyqt5")

    # 改用标签上色
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
