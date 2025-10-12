import argparse
import sys
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
    args.solid = r"E:\CAD数据集\项目\GFR_Dataset_Final\GFR_00196.step"

    print(f"正在加载 STEP 文件: {args.solid}...")
    compound = load_single_compound_from_step(args.solid)
    if compound is None:
        print(f"错误：无法加载 STEP 文件 '{args.solid}'。文件可能已损坏或格式不受支持。")
        sys.exit(1)

    # 在对任何部分进行操作前，先进行缩放
    compound = compound.scale_to_unit_box()

    entity_to_process = None

    # 1. 优先寻找 Solid
    solids = list(compound.solids())
    if solids:
        print(f"成功找到 {len(solids)} 个 Solid 实体。将处理第一个。")
        entity_to_process = solids[0]
    else:
        # 2. 如果没有 Solid，则寻找 Shell
        print("未找到 Solid。正在尝试寻找 Shell...")
        shells = list(compound.shells())
        if shells:
            print(f"成功找到 {len(shells)} 个 Shell 实体。将处理第一个。")
            entity_to_process = shells[0]
        else:
            # 3. 如果连 Shell 都没有，则报错退出
            print("错误：在文件中既没有找到 Solid 也没有找到 Shell。无法构建面邻接图。")
            print("请检查 STEP 文件，它可能只包含离散的面或线，或者是一个空的 Compound。")
            # 也可以尝试直接用 compound.faces() 获取所有面进行可视化，但无法建立邻接关系
            sys.exit(1)
    
    # 现在 entity_to_process 要么是一个 Solid，要么是一个 Shell
    graph = build_graph(entity_to_process, 10, 10, 10)

    v = Viewer(backend="pyqt5")

    


    print([attr for attr in dir(v) if not attr.startswith("__")])


    title = "Face Labels Visualization By Dyk"



    # --- 绘制所有面（可选）
    display_all_faces(
        v, 
        graph.face_obj_list, 
        highlight_indices = [149,220,236,240],      # 这里可以指定要高亮的面编号，如 [2,5]
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
