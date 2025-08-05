import json
import copy
import time
import math
import random
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import networkx as nx
from networkx.readwrite import json_graph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from tqdm import tqdm


class GetLabelMatrix:
    def __init__(self,root_dir,split="train",):
        return

    def load_graph_body(self,body):
        """Load a graph created from a brep body"""
        body_json_file = root_dir / f"{body}.json"
        # Graph json that we feed in to make the graph
        g_json = self.load_graph_json_data(body_json_file)
        # Copy of the graph data
        gd = {"nodes": [], "properties": g_json["properties"]}
        # Center and scale if needed
        bbox = g_json["properties"]["bounding_box"]
        center = self.get_center_from_bounding_box(bbox)

        # Get the joint node indices that serve as labels
        face_count = g_json["properties"]["face_count"]
        edge_count = g_json["properties"]["edge_count"]
        node_count = len(g_json["nodes"])
        link_count = len(g_json["links"])
        # Throw out graphs without edges
        if node_count < 2:
            return None, None, face_count, edge_count, body_json_file
        if link_count <= 0:
            return None, None, face_count, edge_count, body_json_file

        for index, node in enumerate(g_json["nodes"]):
            self.copy_features(node, gd)
            if "is_degenerate" in node and node["is_degenerate"]:
                # If we have a degenerate edge we give some default values
                node["x"] = torch.zeros((self.grid_size, self.grid_size, self.grid_channels))
                node["entity_types"], _ = self.get_node_entity_type(node)
                node["is_face"] = torch.tensor(0, dtype=torch.long)
                node["area"] = torch.tensor(0, dtype=torch.float)
                node["length"] = torch.tensor(0, dtype=torch.float)
                node["face_reversed"] = torch.tensor(0, dtype=torch.long)
                node["edge_reversed"] = torch.tensor(0, dtype=torch.long)
                node["reversed"] = torch.tensor(0, dtype=torch.long)
                node["convexity"] = self.get_node_convexity(node)
                node["dihedral_angle"] = torch.tensor(0, dtype=torch.long)
            else:
                # Pull out the node features
                node["x"] = self.get_grid_features(node, center)
                # Pull out the surface or curve type
                node["entity_types"], _ = self.get_node_entity_type(node)
                # Feature indicating if this node is a B-Rep face
                node["is_face"] = torch.tensor(int("surface_type" in node), dtype=torch.long)
                # Feature for the area or length of the entity
                node["area"] = self.get_node_area(node)
                node["length"] = self.get_node_length(node)
                # Separate features for faces and edge reversal
                face_reversed, edge_reversed = self.get_node_face_edge_reversed(node)
                node["face_reversed"] = face_reversed
                node["edge_reversed"] = edge_reversed
                # Combined feature indicating if the face/edge is reversed
                node["reversed"] = torch.tensor(int(node["reversed"]), dtype=torch.long)
                # Edge convexity feature
                node["convexity"] = self.get_node_convexity(node)
                # Dihedral Angle
                node["dihedral_angle"] = self.get_node_dihedral_angle(node)

            # Radius
            node["radius"] = self.get_node_radius(node)
            # Remove the grid node features to save extra copying
            # as we have already pulled out what we need
            self.delete_features(node)

        # Load the networkx graph file
        nxg = json_graph.node_link_graph(g_json)
        # Convert to a graph
        g = from_networkx(nxg)
        g = self.reshape_graph_features(g)
        # Check we have the same number of nodes and edges
        assert nxg.number_of_edges() == g.num_edges
        assert nxg.number_of_nodes() == g.num_nodes
        return g, gd, face_count, edge_count, body_json_file


    def load_graph(self, joint_file_name):
        """Load a joint file and return a graph"""
        joint_file = self.root_dir / joint_file_name
        with open(joint_file, encoding="utf8") as f:
            joint_data = json.load(f)
        g1, g1d, face_count1, edge_count1, g1_json_file = self.load_graph_body(
            joint_data["body_one"])
        if g1 is None:
            return None
        g2, g2d, face_count2, edge_count2, g2_json_file = self.load_graph_body( # 
            joint_data["body_two"])
        if g2 is None:
            return None
        # Limit the maximum number of combined nodes
        total_nodes = face_count1 + edge_count1 + face_count2 + edge_count2
        if self.max_node_count > 0:
            if total_nodes > self.max_node_count:
                return None
        # Get the joint label matrix
        label_matrix = self.get_label_matrix(
            joint_data,
            g1, g2,
            g1d, g2d,
            face_count1, face_count2,
            edge_count1, edge_count2
        )
        # Create the joint graph from the label matrix
        joint_graph = self.make_joint_graph(g1, g2, label_matrix)
        # Scale geometry features from both graphs with a common scale
        if self.center_and_scale:
            scale_good = self.scale_geometry(
                g1.x,
                g2.x,
                area_features1=g1.area,
                area_features2=g2.area,
                length_features1=g1.length,
                length_features2=g2.length,
            )
            # Throw out if we can't scale properly due to the masked surface area
            if not scale_good:
                print("Discarding graph with bad scale")
                return None
        # Flag to indicate if this design has holes
        holes = joint_data.get("holes", [])
        has_holes = len(holes) > 0
        # Transforms for this joint set
        transforms = self.get_joint_transforms(joint_data)
        return g1, g2, joint_graph, joint_file, g1_json_file, g2_json_file, has_holes, transforms


    def get_label_matrix(self, joint_data, g1, g2, g1d, g2d, face_count1, face_count2, edge_count1, edge_count2):
        """Get the label matrix containing user selected entities and various label augmentations"""
        joints = joint_data["joints"]
        holes = joint_data.get("holes", [])
        entity_count1 = face_count1 + edge_count1
        entity_count2 = face_count2 + edge_count2
        # Labels are as follows:
        # 0 - Non joint
        # 1 - Joints (selected by user)
        # 2 - Ambiguous joint
        # 3 - Joint equivalents
        # 4 - Ambiguous joint equivalents
        # 5 - Hole
        # 6 - Hole equivalents
        label_matrix = torch.zeros((entity_count1, entity_count2), dtype=torch.long)
        for joint in joints:
            entity1 = joint["geometry_or_origin_one"]["entity_one"]
            entity1_index = entity1["index"]
            entity1_type = entity1["type"]
            entity2 = joint["geometry_or_origin_two"]["entity_one"]
            entity2_index = entity2["index"]
            entity2_type = entity2["type"]
            # Offset the joint indices for use in the label matrix
            entity1_index = self.offset_joint_index(
                entity1_index, entity1_type, face_count1, entity_count1)
            entity2_index = self.offset_joint_index(
                entity2_index, entity2_type, face_count2, entity_count2)
            # Set the joint equivalent indices
            eq1_indices = self.get_joint_equivalents(
                joint["geometry_or_origin_one"], face_count1, entity_count1)
            eq2_indices = self.get_joint_equivalents(
                joint["geometry_or_origin_two"], face_count2, entity_count2)
            # Add the actual entities
            eq1_indices.append(entity1_index)
            eq2_indices.append(entity2_index)
            # For every pair we set a joint
            for eq1_index in eq1_indices:
                for eq2_index in eq2_indices:
                    # Only set non-joints, we don't want to replace other labels
                    if label_matrix[eq1_index][eq2_index] == self.label_map["Non-joint"]:
                        label_matrix[eq1_index][eq2_index] = self.label_map["JointEquivalent"]
            # Set the user selected joint indices
            label_matrix[entity1_index][entity2_index] = self.label_map["Joint"]

        # Include ambiguous and hole labels
        # Adding separate labels to the label_matrix
        # We need to do this after all joints are marked out as labels
        g1_ambiguous, g2_ambiguous = self.set_ambiguous_labels(g1, g2, label_matrix)
        g1_holes, g2_holes = self.set_hole_labels(g1d, g2d, label_matrix, joint_data)

        # Only do further work if we have holes or ambiguous entities
        eq_count = len(g1_ambiguous) + len(g2_ambiguous) + len(g1_holes) + len(g2_holes)
        if eq_count > 0:
            # First calculate the axis lines and cache them
            g1_axis_lines = self.get_axis_lines_from_graph(g1d)
            g2_axis_lines = self.get_axis_lines_from_graph(g2d)

            # Now find and set the equivalents
            self.set_equivalents(
                g1_ambiguous, g2_ambiguous,
                g1_axis_lines, g2_axis_lines,
                label_matrix, self.label_map["AmbiguousEquivalent"]
            )
            self.set_equivalents(
                g1_holes, g2_holes,
                g1_axis_lines, g2_axis_lines,
                label_matrix, self.label_map["HoleEquivalent"]
            )
        return label_matrix



    def make_joint_graph(self, graph1, graph2, label_matrix):
        """Create a joint graph connecting graph1 and graph2 densely"""
        nodes_indices_first_graph = torch.arange(graph1.num_nodes)
        # We want to treat both graphs as one, so order the indices of the second graph's nodes
        # sequentially after the first graph's node indices
        nodes_indices_second_graph = torch.arange(graph2.num_nodes) + graph1.num_nodes
        edges_between_graphs_1 = torch.cartesian_prod(nodes_indices_first_graph, nodes_indices_second_graph).transpose(1, 0)
        # Pradeep: turn these on of we want bidirectional edges among the bodies
        # edges_between_graphs_2 = torch.cartesian_prod(nodes_indices_second_graph, nodes_indices_first_graph).transpose(1, 0)
        # edges_between_graphs = torch.cat((edges_between_graphs_1, edges_between_graphs_2), dim=0)
        num_nodes = graph1.num_nodes + graph2.num_nodes
        empty = torch.zeros((num_nodes, 1), device=graph1.x.device)
        joint_graph = Data(x=empty, edge_index=edges_between_graphs_1)
        joint_graph.num_nodes = num_nodes
        joint_graph.edge_attr = label_matrix.view(-1)
        joint_graph.num_nodes_graph1 = graph1.num_nodes
        joint_graph.num_nodes_graph2 = graph2.num_nodes
        return joint_graph


    def get_all_joint_files(root_dir):
        """Get all the json joint files that look like joint_set_00025.json"""
        pattern = "joint_set_[0-9][0-9][0-9][0-9][0-9].json"
        return [f.name for f in Path(root_dir).glob(pattern)]

    def get_joint_files():
        """Get the joint files to load"""
        all_joint_files = get_all_joint_files()
        # Create the train test split
        joint_files = get_split(all_joint_files)
        # Using only a subset of files
        if self.limit > 0:
            joint_files = joint_files[:self.limit]
        # Store the original file count
        # to keep track of the number of files we filter
        # from the official train/test split
        original_file_count = len(joint_files)
        print(f"Loading {len(joint_files)} {self.split} data")
        return joint_files


    def get_split(self, all_joint_files):
        """Get the train/test split"""
        # First check if we have the official split in the dataset dir
        split_file = self.root_dir / "train_test.json"
        # Look in the parent directory too if we can't find it
        if not split_file.exists():
            split_file = self.root_dir.parent / "train_test.json"
        if split_file.exists():
            print("Using official train test split")
            train_joints = []
            val_joints = []
            test_joints = []
            with open(split_file, encoding="utf8") as f:
                official_split = json.load(f)
            if self.split == "train":
                joint_files = official_split["train"]
            elif self.split == "val" or self.split == "validation":
                joint_files = official_split["validation"]
            elif self.split == "test":
                joint_files = official_split["test"]
            elif self.split == "mix_test":
                if "mix_test" not in official_split:
                    raise Exception("Mix test split missing")
                else:
                    joint_files = official_split["mix_test"]
            elif self.split == "all":
                joint_files = []
                for split_files in official_split.values():
                    joint_files.extend(split_files)
            else:
                raise Exception("Unknown split name")
            joint_files = [f"{f}.json" for f in joint_files]
            if self.shuffle_split:
                random.Random(self.seed).shuffle(joint_files)
            return joint_files
        else:
            # We don't have an official split, so we make one
            print("Using new train test split")
            if self.split != "all":
                trainval_joints, test_joints = train_test_split(
                    all_joint_files, test_size=0.2, random_state=self.seed,
                )
                train_joints, val_joints = train_test_split(
                    trainval_joints, test_size=0.25, random_state=self.seed + self.seed,
                )
            if self.split == "train":
                joint_files = train_joints
            elif self.split == "val" or self.split == "validation":
                joint_files = val_joints
            elif self.split == "test":
                joint_files = test_joints
            elif self.split == "all":
                joint_files = all_joint_files
            else:
                raise Exception("Unknown split name")
            return joint_files




def main():
    return 




if __name__ == "__main__":
    main()
