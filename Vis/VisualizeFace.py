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
    """
    Load data from a STEP file as a single compound
    """
    return Compound.load_from_step(step_filename)

def load_step(step_filename):
    """
    Load solids from a STEP file
    """
    compound = load_single_compound_from_step(step_filename)
    return list(compound.solids())

def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    # Compute the UV-grids for faces
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

    # Compute the U-grids for edges
    graph_edge_feat = []
    for edge_idx in graph.edges:
        edge = graph.edges[edge_idx]["edge"]
        if not edge.has_curve():
            continue
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    # 关键: 保存原始 graph 用于查face
    dgl_graph.nodes_dict = graph.nodes
    return dgl_graph

def color_faces(solid, graph, viewer, face_indices, color=(1, 0, 0)):
    """
    给选中的面上色
    :param solid: occwl.Solid
    :param graph: dgl graph, graph.nodes_dict 里保存了每个face
    :param viewer: occwl.Viewer
    :param face_indices: 需要上色的face编号列表
    :param color: RGB, 默认红色
    """
    for face_idx in face_indices:
        # 兼容graph.nodes_dict和graph.nodes
        nodes_dict = getattr(graph, "nodes_dict", None)
        if nodes_dict is None:
            try:
                nodes_dict = graph.nodes
            except:
                nodes_dict = graph._graph.nodes
        # 找到对应的face对象
        face = nodes_dict[face_idx]["face"]
        viewer.display(face, color=color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Visualize and color selected faces of BRep solid"
    )
    parser.add_argument("--solid", type=str, help="Solid STEP file")
    args = parser.parse_args()

    # 手动设定STEP路径（如需命令行参数，删除这行）
    args.solid = r'D:\CAD数据集\j1.0.0\joint\step\137295_a2385b57_0000_1.step'

    solid = load_step(args.solid)[0]
    solid = solid.scale_to_unit_box()
    graph = build_graph(solid, 10, 10, 10)

    v = Viewer(backend="pyqt5")
    # 1. 先显示整体
    v.display(solid)

    # 2. 给选定的face上色（比如13号面，红色）
    color_faces(solid, graph, viewer=v, face_indices=[13], color=(1, 0, 0))  # 红色
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
    """
    Load data from a STEP file as a single compound
    """
    return Compound.load_from_step(step_filename)

def load_step(step_filename):
    """
    Load solids from a STEP file
    """
    compound = load_single_compound_from_step(step_filename)
    return list(compound.solids())

def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    # Compute the UV-grids for faces
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

    # Compute the U-grids for edges
    graph_edge_feat = []
    for edge_idx in graph.edges:
        edge = graph.edges[edge_idx]["edge"]
        if not edge.has_curve():
            continue
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    # 关键: 保存原始 graph 用于查face
    dgl_graph.nodes_dict = graph.nodes
    return dgl_graph

def color_faces(solid, graph, viewer, face_indices, color=(1, 0, 0)):
    """
    给选中的面上色
    :param solid: occwl.Solid
    :param graph: dgl graph, graph.nodes_dict 里保存了每个face
    :param viewer: occwl.Viewer
    :param face_indices: 需要上色的face编号列表
    :param color: RGB, 默认红色
    """
    for face_idx in face_indices:
        # 兼容graph.nodes_dict和graph.nodes
        nodes_dict = getattr(graph, "nodes_dict", None)
        if nodes_dict is None:
            try:
                nodes_dict = graph.nodes
            except:
                nodes_dict = graph._graph.nodes
        # 找到对应的face对象
        face = nodes_dict[face_idx]["face"]
        viewer.display(face, color=color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Visualize and color selected faces of BRep solid"
    )
    parser.add_argument("--solid", type=str, help="Solid STEP file")
    args = parser.parse_args()

    # 手动设定STEP路径（如需命令行参数，删除这行）
    args.solid = "D:\CAD数据集\j1.0.0\joint\step\\7778_3a9748b3_0025_1.step"

    solid = load_step(args.solid)[0]
    solid = solid.scale_to_unit_box()
    graph = build_graph(solid, 10, 10, 10)

    v = Viewer(backend="pyqt5")
    # 1. 先显示整体
    v.display(solid)

    # 2. 给选定的face上色（比如13号面，红色）
    color_faces(solid, graph, viewer=v, face_indices=[1,2,3,4], color=(0, 1, 1) ) # 青色

    v.fit()
    v.show()

    v.fit()
    v.show()
