import argparse
import numpy as np
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
    """
    将所有面绘制到viewer，高亮highlight_indices对应的面
    """
    highlight_indices = highlight_indices or []
    for idx, face in enumerate(face_obj_list):
        if idx in highlight_indices:
            viewer.display(face, color=highlight_color)
        else:
            viewer.display(face, color=default_color)

def display_edge_with_fake_width(viewer, edge, color=(1,0,0), width=12, n_sample=80):
    """
    用打点方式伪造粗边
    :param viewer: occwl.Viewer
    :param edge: occwl.edge.Edge
    :param color: RGB
    :param width: 点大小
    :param n_sample: 采样点数
    """
    points = ugrid(edge, method="point", num_u=n_sample)
    viewer.display_points(points, color=color, marker="point", scale=width)

def display_all_edges(viewer, edge_obj_list, highlight_indices=None, highlight_color=(0,1,0), default_color=(0.2,0.2,0.2), fake_width=14):
    """
    将所有边绘制到viewer，高亮highlight_indices对应的边（粗线高亮）
    """
    highlight_indices = highlight_indices or []
    for idx, edge in enumerate(edge_obj_list):
        if idx in highlight_indices:
            viewer.display(edge, color=highlight_color)  # 原生线
            display_edge_with_fake_width(viewer, edge, color=highlight_color, width=fake_width, n_sample=100)
        else:
            viewer.display(edge, color=default_color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Visualize and color selected edges of BRep solid"
    )
    parser.add_argument("--solid", type=str, help="Solid STEP file")
    args = parser.parse_args()

    # 如果你要传命令行参数，就注释掉下面这行
    args.solid = 'C:/Users/20268/Desktop/demo/00670180.step'

    solid = load_step(args.solid)[0]
    solid = solid.scale_to_unit_box()
    graph = build_graph(solid, 10, 10, 10)

    v = Viewer(backend="pyqt5")

    # --- 绘制所有面（可选）
    display_all_faces(
        v, 
        graph.face_obj_list, 
        highlight_indices=[],      # 这里可以指定要高亮的面编号，如 [2,5]
        highlight_color=(0,1,0), 
        default_color=(0.8,0.8,0.8)
    )

    # --- 绘制所有边，高亮[5,3,1]号边
    display_all_edges(
        v, 
        graph.edge_obj_list, 
        highlight_indices=[], 
        highlight_color=(1,0,0),   # 红色
        default_color=(0.3,0.3,0.3),
        fake_width=18               # 你想要的粗度
    )

    v.fit()
    v.show()
