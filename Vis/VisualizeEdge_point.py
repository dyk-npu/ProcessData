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
from dgl.data.utils import load_graphs

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
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    # Compute the UV-grids for faces
    graph_face_feat = []
    for face_idx in graph.nodes:
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
    for edge_idx in graph.edges:
        # Get the B-rep edge
        edge = graph.edges[edge_idx]["edge"]
        # Ignore dgenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        # Concatenate channel-wise to form edge feature tensor
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    # Convert face-bin_global graph to DGL format
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    return dgl_graph


def draw_edge_uvgrids(solid, graph, viewer,num_edge):

    for edge_idx in num_edge:
        a_face_data = graph.edata["x"][edge_idx]
        edge_uvgrids = a_face_data.view(-1, 6)
        points = []
        tangents = []
        for idx in range(edge_uvgrids.shape[0]):
            points.append(edge_uvgrids[idx, :3].cpu().numpy())
            tangents.append(edge_uvgrids[idx, 3:6].cpu().numpy())

        points = np.asarray(points, dtype=np.float32)
        tangents = np.asarray(tangents, dtype=np.float32)

        bbox = solid.box()
        max_length = max(bbox.x_length(), bbox.y_length(), bbox.z_length())

        # Draw the points
        viewer.display_points(points, color=(1, 0, 1), marker="point", scale=3*max_length)

        # Draw the tangents
        # for pt, tgt in zip(points, tangents):
        #    viewer.display(Edge.make_line_from_points(pt, pt + tgt * 0.1 * max_length), color=(1, 0, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Visualize UV-grids and face bin_global graphs for testing"
    )
    parser.add_argument("--solid", type=str, help="Solid STEP file")
    args = parser.parse_args()

    args.solid = "../Data/MFTR/steps/20240116_231044_100_result.step"

    solid = load_step(args.solid)[0]
    solid = solid.scale_to_unit_box()
    graph = build_graph(solid, 10, 10, 10)

    v = Viewer(backend="wx")
    # Draw the solid
    v.display(solid)
    # Draw the face UV-grids
    draw_edge_uvgrids(solid, graph, viewer = v,num_edge = [5,3,1])

    v.fit()
    v.show()