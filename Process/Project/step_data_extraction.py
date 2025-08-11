#-------------------------------------------------------
# -*- coding: utf-8 -*-
#-------------------------------------------------------
import gc
import json
import os
import sys
import random
import time
import math
import pickle
import shutup
import numpy as np
#
import argparse
#
from tqdm import tqdm
from functools import partial
from pathlib import Path
from itertools import repeat,accumulate
from multiprocessing.pool import Pool
from collections import defaultdict
#
import networkx as nx
#
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
#
#-------------------------------------------------------
#
#-------------------------------------------------------
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.uvgrid import uvgrid, ugrid
from occwl.edge import Edge
from occwl.shell import Shell
from occwl.face import Face
from occwl.shape import Shape
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
#
#------------------------------------------------------
from OCC.Core.BRep import BRep_Tool
from OCC.Extend import TopologyUtils
from OCC.Extend.TopologyUtils import WireExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, BRepGProp_Face
from OCC.Core.BRepCheck import BRepCheck_Analyzer
#
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse,BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Section
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.gp import gp_Trsf, gp_Pnt, gp_Ax1, gp_Vec, gp_Dir
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs,STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Quantity import Quantity_NOC_RED, Quantity_NOC_GREEN, Quantity_NOC_BLUE, Quantity_NOC_YELLOW ,Quantity_Color
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE,TopAbs_SHAPE, TopAbs_WIRE
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeWire
from OCC.Core.BRepFill import BRepFill_Filling
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Edge, topods_Wire, topods_Edge, topods_Vertex, topods_Face, TopoDS_Shape
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
                              GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
                              GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,GeomAbs_SurfaceType)
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#
shutup.please()
#
#------------------------------------------------------
#------------------------------------------------------
def load_one_graph(graph_data):
    #
    pyg_graph = Data()
    pyg_graph.graph = graph_data['graph']
    #
    pyg_graph.node_data = torch.tensor(graph_data["graph_face_grid"],dtype=torch.float32).permute(0,2,3,1) 
    pyg_graph.edge_data = torch.tensor(graph_data["graph_edge_grid"], dtype=torch.float32).permute(0, 2, 1)  
    #
    graph_face_attr = torch.tensor(graph_data["graph_face_attr"], dtype=torch.float32) #[num_nodes,feature])
    node_type = graph_face_attr[: , 0:9]        #one-hot list
    node_area = graph_face_attr[: , 9]          # area value
    node_rational = graph_face_attr[: , 10]     # if or not rational
    node_centroid = graph_face_attr[: , 11:14]  # coordinates of center point
    #
    pyg_graph.face_type = node_type
    pyg_graph.face_area = node_area
    pyg_graph.face_rational = node_rational
    pyg_graph.face_centroid = node_centroid
    #数据标签待更新
    pyg_graph.label_feature = torch.tensor(graph_data["label_feature"], dtype=torch.float32)  # label_feature[num_nodes] CAD的label
    #
    graph_edge_attr = torch.tensor(graph_data["graph_edge_attr"], dtype=torch.float32)
    edge_convexity = graph_edge_attr[: , 0:3]
    edge_length = graph_edge_attr[: , 3]
    edge_type = graph_edge_attr[: , 4:15]
    #
    pyg_graph.edge_type = edge_type
    pyg_graph.edge_length = edge_length
    pyg_graph.edge_convexity = edge_convexity
    #
    num_nodes = graph_data['simple_graph']['num_nodes']
    edges = torch.tensor(graph_data['simple_graph']['edges'])
    #
    graph = Data(edge_index=edges,num_nodes=num_nodes)
    G = to_networkx(graph,edge_attrs=None) 
    dense_adj = torch.tensor(nx.adjacency_matrix(G).todense()).type(torch.float32)
    pyg_graph.node_degree = dense_adj.long().sum(dim=1).view(-1)
    pyg_graph.attn_bias = torch.zeros([num_nodes + 1, num_nodes + 1], dtype=torch.float32)
    #
    pyg_graph.angle = graph_data["angle_matrix"]  # angle[num_nodes, num_nodes]
    pyg_graph.centroid_distance = graph_data["centroid_distance_matrix"]  # centroid_distance[num_nodes, num_nodes]
    pyg_graph.shortest_distance = graph_data["shortest_distance_matrix"].to(torch.int32)  # shortest_distance[num_nodes, num_nodes]
    pyg_graph.edge_path = graph_data["edge_path_matrix"].to(torch.int32)  # edge_path[num_nodes, num_nodes, max_distance]
    #
    return pyg_graph
#------------------------------------------------------
#-------------------------------------------------------
def load_body_from_step(step_file):
    """
    Load the body from the step file.  
    We expect only one body in each file
    """
    assert Path(step_file).suffix in ['.step', '.stp', '.STEP', '.STP']
    reader = STEPControl_Reader()
    reader.ReadFile(str(step_file))
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape
#----------------------------------------------------------
#----------------------------------------------------------
def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)
#----------------------------------------------------------
#----------------------------------------------------------
def save_json_data(path_name, data):
    """Export a data to a json file"""
    with open(path_name, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)
#----------------------------------------------------------
#----------------------------------------------------------
"""
The entity mapper allows you to map between occwl entities and integer
identifiers which can be used as indices into arrays of feature vectors
or the rows and columns of incidence matrices. 

NOTE:  
    
    Only oriented edges which are used by wires are included in the oriented 
    edge map.  In the case of edges which are open (i.e. they are adjacent
    to a hole in the solid), only one oriented edge is present. Use the function
    
    EntityMapper.oriented_edge_exists(oriented_edge)
    
    to check if an oriented edge is used by a wire and known to the entity mapper.
"""
#
#
#----------------------------------------------------------
#----------------------------------------------------------
class EntityMapper:
    """
    This class allows us to map between occwl entities and integer
    identifiers which can be used as indices into arrays of feature vectors
    or the rows and columns of incidence matrices. 
    """
    
    def __init__(self, solid):
        """
        Create a mapper object for solid
        
        Args:
            
            solid (occwl.solid.Solid): A single solid
        """
        
        # Create the dictionaries which will map the
        # objects hash values to the indices
        self.face_map = dict()
        self.face_index_map = dict()
        self.face_edge_map = dict()
        self.edge_index_map = dict()
        self.edge_index_edge = dict()
        self.wire_map = dict()
        self.edge_map = dict()
        self.oriented_edge_map = dict()
        self.vertex_map = dict()
        self.vertex_index_map = dict()
        self.edge_vertex_map = dict()

        # Build the index lookup tables
        self._append_faces(solid)
        self._append_wires(solid)
        self._append_edges(solid)
        self._append_oriented_edges(solid)
        self._append_vertices(solid)

    # The following functions are the interface for
    # users of the class to access the indices
    # which will reptresent the Open Cascade entities
    
    def get_num_edges(self):
        return len(self.edge_map)
    
    def get_num_faces(self):
        return len(self.face_map)
    
    def face_index(self, face):
        """
        Find the index of a face
        """
        return self.face_map[self._get_hash(face)]
    
    def wire_index(self, wire):
        """
        Find the index of a wire
        """
        return self.wire_map[self._get_hash(wire)]
    
    def edge_index(self, edge):
        """
        Find the index of an edge
        """
        return self.edge_map[self._get_hash(edge)]
    
    def oriented_edge_index(self, oriented_edge):
        """
        Find the index of a oriented edge.  i.e. a coedge
        """
        tup = (self._get_hash(oriented_edge), oriented_edge.Reversed())
        return self.oriented_edge_map[tup]
    
    def oriented_edge_exists(self, oriented_edge):
        tup = (self._get_hash(oriented_edge), oriented_edge.Reversed())
        return tup in self.oriented_edge_map
    
    def vertex_index(self, vertex):
        """
        Find the index of a vertex
        """
        return self.vertex_map[self._get_hash(vertex)]
    
    # These functions are used internally to build the map
    
    def _get_hash(self, ent):
        return ent.__hash__()
    
    def _append_faces(self, solid):
        explorer = TopExp_Explorer(solid, TopAbs_FACE)
        while explorer.More():
            face = topods_Face(explorer.Current())
            h = self._get_hash(face)
            self.face_map[h] = len(self.face_map)
            self.face_index_map[self.face_map[h]] = face
            explorer_face = TopExp_Explorer(face, TopAbs_EDGE)
            self.face_edge_map[h]=[]
            while explorer_face.More():
                edge = topods_Edge(explorer_face.Current())
                h_edge = self._get_hash(edge)
                if h_edge not in self.face_edge_map[h]:
                    self.face_edge_map[h].append(h_edge)
                #
                vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
                self.edge_vertex_map[h_edge]=[]
                while vertex_explorer.More():
                    vertex = topods_Vertex(vertex_explorer.Current())
                    h_vertex = self._get_hash(vertex)
                    if h_vertex not in self.edge_vertex_map[h_edge]:
                        self.edge_vertex_map[h_edge].append(h_vertex)
                    #
                    vertex_explorer.Next()
                explorer_face.Next()
            explorer.Next()
        #
    def _append_wires(self, solid):
        explorer = TopExp_Explorer(solid, TopAbs_WIRE)
        while explorer.More():
            wire = topods_Wire(explorer.Current())
            h = self._get_hash(wire)
            self.wire_map[h] = len(self.wire_map)
            explorer.Next()
    #
    def _append_edges(self, solid):
        explorer = TopExp_Explorer(solid, TopAbs_EDGE)
        while explorer.More():
            edge = topods_Edge(explorer.Current())
            h = self._get_hash(edge)
            self.edge_map[h] = len(self.edge_map)
            self.edge_index_edge[self.edge_map[h]]=edge
            self.edge_index_map[h]=edge
            explorer.Next()
    #
    def _append_oriented_edges(self, solid):
        wire_explorer = TopExp_Explorer(solid, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods_Wire(wire_explorer.Current())
            wire_exp = WireExplorer(wire)
            # 遍历Wire中的有向边（假设ordered_edges()方法可用）
            for oriented_edge in list( wire_exp.ordered_edges()):
                h = self._get_hash(oriented_edge)
                tup = (h, oriented_edge.Reversed())
                if tup not in self.oriented_edge_map:
                    self.oriented_edge_map[tup] = len(self.oriented_edge_map)
            wire_explorer.Next()
    #
    def _append_vertices(self, solid):
        explorer = TopExp_Explorer(solid, TopAbs_VERTEX)
        while explorer.More():
            vertex = topods_Vertex(explorer.Current())
            h = self._get_hash(vertex)
            self.vertex_map[h] = len(self.vertex_map)
            self.vertex_index_map[self.vertex_map[h]]=vertex
            explorer.Next()
    #
#----------------------------------------------------------
#----------------------------------------------------------
def face_adjacency(mapper, self_loops=False):
    """
    Creates a face adjacency graph from the given shape (Solid or Compound)
    
    Args:
        shape (Shell, Solid, or Compound): Shape
        self_loops (bool, optional): Whether to add self loops in the graph. Defaults to False.
        
    Returns:
        nx.DiGraph: Each B-rep face is mapped to a node with its index and each B-rep edge is mapped to an edge in the graph
                    Node attributes:
                    - "face": contains the B-rep face
                    Edge attributes:
                    - "edge": contains the B-rep (ordered) edge
                    - "edge_idx": index of the (ordered) edge in the solid
        None: if the shape is non-manifold or open
    """
    #assert isinstance(shape, (Shell, Solid, Compound))
    #EntityMapper 把shape里面实体hash值跟有序的序号建立一一对应的关系；
    #mapper = EntityMapper(shape)
    graph = nx.DiGraph()
    #
    
    face_index_map = mapper.face_index_map
    get_item = face_index_map.__getitem__
    for key in face_index_map:
        value=get_item(key) #value 为 face_occ
        face_idx = key
        graph.add_node(face_idx, face=value)
    #
    edge_face_map={}
    #
    face_edge_map=mapper.face_edge_map
    get_item = face_edge_map.__getitem__
    for key in face_edge_map:
        value=get_item(key)
        iter_list = value.__iter__()
        try:
            while True:
                item = next(iter_list)
                if item not in edge_face_map.keys():
                    edge_face_map[item]=[]
                if key not in edge_face_map[item]:
                    edge_face_map[item].append(key)
        except:
            pass
    #
    tmp_list=[]
    get_item = edge_face_map.__getitem__
    for key in edge_face_map:
        value=get_item(key)
        if len(value) ==0:
            tmp_list.append(key)
        if len(value) ==1:
            edge_face_map[key].append(value[0])
    for key in tmp_list:
        edge_face_map.pop(key)
    #
    for key in edge_face_map.keys():
        left_index = mapper.face_index(edge_face_map[key][0])
        right_index = mapper.face_index(edge_face_map[key][1])
        edge = mapper.edge_index_map[key]
        edge_idx = mapper.edge_index(edge)
        graph.add_edge(left_index, right_index, edge=edge, edge_index=edge_idx) 
    #
    return graph
#----------------------------------------------------------
#----------------------------------------------------------
def vertex_adjacency(mapper, self_loops=False):
    """ 
    Creates a vertex adjacency graph from the given shape (Wire, Solid or Compound)
    
    Args:
        shape (Wire, Face, Shell, Solid, or Compound): Shape
        self_loops (bool, optional): Whether to add self loops in the graph. Defaults to False.
    
    Returns:
        nx.DiGraph: Each B-rep vertex is mapped to a node with its index and each B-rep edge is mapped to an edge in the graph
                    Node attributes:
                    - "vertex": contains the B-rep vertex
                    Edge attributes:
                    - "edge": contains the B-rep (ordered) edge
                    - "edge_idx": index of the (ordered) edge in the solid
    """
    #assert isinstance(shape, (Wire, Face, Shell, Solid, Compound))
    #mapper = EntityMapper(shape)
    graph = nx.DiGraph()
    #
    vertex_index_map = mapper.vertex_index_map
    get_item = vertex_index_map.__getitem__
    for key in vertex_index_map:
        value=get_item(key)
        graph.add_node(key, vertex=value)
    #
    edge_index_edge = mapper.edge_index_edge
    edge_map = mapper.edge_map
    edge_vertex_map = mapper.edge_vertex_map
    get_item = edge_vertex_map.__getitem__
    for key in edge_vertex_map:
        value=get_item(key)
        edge_idx = edge_map[key]
        edge = edge_index_edge[edge_idx]
        if len(value) ==0:
            pass
        if len(value) ==1:
            graph.add_edge(value[0], value[0], edge=edge, edge_index=edge_idx)
        if len(value) ==2:
            graph.add_edge(value[0], value[1], edge=edge, edge_index=edge_idx)
    #
    return graph
#----------------------------------------------------------
#----------------------------------------------------------
class TopologyChecker:
    # modified from BREPNET: https://github.com/AutodeskAILab/BRepNet/blob/master/pipeline/extract_brepnet_data_from_step.py
    def __init__(self):
        pass
    #
    def find_edges_from_wires(self, top_exp):
        edge_set = set()
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_set.add(edge)
        return edge_set
    #
    def find_edges_from_top_exp(self, top_exp):
        edge_set = set(top_exp.edges())
        return edge_set
    #
    def check_closed(self, body):
        # In Open Cascade, unlinked (open) edges can be identified
        # as they appear in the edges iterator when ignore_orientation=False
        # but are not present in any wire
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        edges_from_wires = self.find_edges_from_wires(top_exp)
        edges_from_top_exp = self.find_edges_from_top_exp(top_exp)
        missing_edges = edges_from_top_exp - edges_from_wires
        return len(missing_edges) == 0
    #
    def check_manifold(self, top_exp):
        faces = set()
        for shell in top_exp.shells():
            for face in top_exp._loop_topo(TopAbs_FACE, shell):
                if face in faces:
                    return False
                faces.add(face)
        return True
    #
    def check_unique_coedges(self, top_exp):
        coedge_set = set()
        for loop in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                orientation = coedge.Orientation()
                tup = (coedge, orientation)
                # We want to detect the case where the coedges
                # are not unique
                if tup in coedge_set:
                    return False
                coedge_set.add(tup)
        return True
    #
    def __call__(self, body):
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        if top_exp.number_of_faces() == 0:
            print('Empty shape')
            return False
        # OCC.BRepCheck, perform topology and geometricals check
        analyzer = BRepCheck_Analyzer(body)
        if not analyzer.IsValid(body):
            print('BRepCheck_Analyzer found defects')
            return False
        # other topology check
        if not self.check_manifold(top_exp):
            print("Non-manifold bodies are not supported")
            return False
        if not self.check_closed(body):
            print("Bodies which are not closed are not supported")
            return False
        if not self.check_unique_coedges(top_exp):
            print("Bodies where the same coedge is uses in multiple loops are not supported")
            return False
        return True
#----------------------------------------------------------
#----------------------------------------------------------
def plane_attribute(face_occ):
    surf_type = BRepAdaptor_Surface(face_occ).GetType()
    if surf_type == GeomAbs_Plane:
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def cylinder_attribute(face_occ):
    surf_type = BRepAdaptor_Surface(face_occ).GetType()
    if surf_type == GeomAbs_Cylinder:
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def cone_attribute(face_occ):
    surf_type = BRepAdaptor_Surface(face_occ).GetType()
    if surf_type == GeomAbs_Cone:
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def sphere_attribute(face_occ):
    surf_type = BRepAdaptor_Surface(face_occ).GetType()
    if surf_type == GeomAbs_Sphere:
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def torus_attribute(face_occ):
    surf_type = BRepAdaptor_Surface(face_occ).GetType()
    if surf_type == GeomAbs_Torus:
        return 1.0
    return 0.0

'''
if surf_type == GeomAbs_BezierSurface:
            return "bezier"
if surf_type == GeomAbs_BSplineSurface:
    return "bspline"
'''
#----------------------------------------------------------
#----------------------------------------------------------
def bezier_attribute(face_occ):
    surf_type = BRepAdaptor_Surface(face_occ).GetType()
    if surf_type == GeomAbs_BezierSurface:
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def bspline_attribute(face_occ):
    surf_type = BRepAdaptor_Surface(face_occ).GetType()
    if surf_type == GeomAbs_BSplineSurface:
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def revolution_attribute(face_occ):
    surf_type = BRepAdaptor_Surface(face_occ).GetType()
    if surf_type == GeomAbs_SurfaceOfRevolution:
        return 1.0
    return 0.0
    #
#def extrusion_attribute(face):
#    if Face(face).surface_type() == "extrusion":
#        return 1.0
#    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def extrusion_attribute(face_occ):
    """
    判断面是否为拉伸面（Extrusion Surface）
    返回: 1.0（是拉伸面）或 0.0（不是）
    """
    # 创建面适配器 获取面类型
    surf_type = BRepAdaptor_Surface(face_occ).GetType()
    
    # 检查是否为拉伸面
    if surf_type == GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion:
        return 1.0
    return 0.0
    #
#def offset_attribute(face):
#    if Face(face).surface_type() == "offset":
#        return 1.0
#    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def offset_attribute(face_occ):
    """
    判断面是否为偏移面（Offset Surface）
    返回: 1.0（是偏移面）或 0.0（不是）
    
    参数:
        face_occ (TopoDS_Face): 待检测的面
    """
    # 创建面适配器
    surf_adaptor = BRepAdaptor_Surface(face_occ)
    
    # 获取面类型
    surf_type = surf_adaptor.GetType()
    
    # 检查是否为偏移面
    if surf_type == GeomAbs_SurfaceType.GeomAbs_OffsetSurface:
        return 1.0
    return 0.0
    #
#def other_attribute(face):
#    if Face(face).surface_type() == "other":
#        return 1.0
#    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def other_attribute(face_occ):
    """
    判断面是否为非标准类型曲面（无法归类到已知明确类型的曲面）
    返回: 1.0（是特殊类型）或 0.0（标准类型）
    
    参数:
        face_occ (TopoDS_Face): 待检测的面
    """
    # 创建面适配器并获取类型
    adaptor = BRepAdaptor_Surface(face_occ)
    surf_type = adaptor.GetType()
    #
    if surf_type == GeomAbs_SurfaceType.GeomAbs_OtherSurface:
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def area_attribute(face):
    geometry_properties = GProp_GProps()
    brepgprop_SurfaceProperties(face, geometry_properties)
    areaValue = geometry_properties.Mass()
    areaValue = areaValue if areaValue > 0.0 else 0.0
    return areaValue
#----------------------------------------------------------
#----------------------------------------------------------
def rational_nurbs_attribute(face):
    surf = BRepAdaptor_Surface(face)
    if surf.GetType() == GeomAbs_BSplineSurface:
        bspline = surf.BSpline()
    elif surf.GetType() == GeomAbs_BezierSurface:
        bspline = surf.Bezier()
    else:
        bspline = None
    
    if bspline is not None:
        if bspline.IsURational() or bspline.IsVRational():
            return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def centroid_attribute(face):
    mass_props = GProp_GProps()
    brepgprop_SurfaceProperties(face, mass_props)
    gPt = mass_props.CentreOfMass()
    
    return gPt.Coord()
#----------------------------------------------------------
#----------------------------------------------------------
def extract_face_attributes( face_occ,attribute_config) -> list:
    # dim [num_faces, 14]
    #待更新，提高稳定性；
    face_attributes = []
    for attribute in attribute_config["face_attributes"]:
        if attribute == "Plane":
            face_attributes.append(plane_attribute(face_occ))
        elif attribute == "Cylinder":
            face_attributes.append(cylinder_attribute(face_occ))
        elif attribute == "Cone":
            face_attributes.append(cone_attribute(face_occ))
        elif attribute == "Sphere":
            face_attributes.append(sphere_attribute(face_occ))
        elif attribute == "Torus":
            face_attributes.append(torus_attribute(face_occ))
        #elif attribute == "BezierSurface":
        #    face_attributes.append(bezier_attribute(face_occ))
        #elif attribute == "BSplineSurface":
        #    face_attributes.append(bspline_attribute(face_occ))
        elif attribute == "Revolution":
            face_attributes.append(revolution_attribute(face_occ))
        elif attribute == "Extrusion":
            face_attributes.append(extrusion_attribute(face_occ))
        elif attribute == "Offset":
            face_attributes.append(offset_attribute(face_occ))
        elif attribute == "Other":
            face_attributes.append(other_attribute(face_occ))
        elif attribute == "FaceAreaAttribute":
            face_attributes.append(area_attribute(face_occ))
        elif attribute == "RationalNurbsFaceAttribute":
            face_attributes.append(rational_nurbs_attribute(face_occ))
        elif attribute == "FaceCentroidAttribute":
            face_attributes.extend(centroid_attribute(face_occ))
        else:
            assert False, "Unknown face attribute"
    return face_attributes                # dim [num_faces, 14]
#----------------------------------------------------------
#----------------------------------------------------------
def extract_face_grid( face,num_srf_u,num_srf_v) -> np.array:
    """
    Extract a UV-Net point grid from the given face.
    
    Returns a tensor [ 7 x num_pts_u x num_pts_v ]
    
    For each point the values are
        
        - x, y, z (point coords)
        - i, j, k (normal vector coordinates)
        - Trimming mast
        
    """
    points = uvgrid(face, num_srf_u, num_srf_v, method="point")    # dim:(num_u , num_v , 3)
    normals = uvgrid(face, num_srf_u, num_srf_v, method="normal")  # dim:(num_u , num_v , 3)
    mask = uvgrid(face, num_srf_u, num_srf_v, method="inside")     # dim:(num_u , num_v , 1)
    
    # This has shape [ num_pts_u x num_pts_v x 7 ]
    single_grid = np.concatenate([points, normals, mask], axis=2)  # dim:(num_u , num_v , 7)
    
    return np.transpose(single_grid, (2, 0, 1))                    # dim:(7 , num_u , num_v )
    #
#----------------------------------------------------------
#----------------------------------------------------------
def find_edge_convexity(edge, faces):
    edge_data = EdgeDataExtractor(Edge(edge),faces, use_arclength_params=False)
    if not edge_data.good:
        # This is the case where the edge is a pole of a sphere
        #print("edge data not good")
        return 0.0
    angle_tol_rads = 0.0872664626  # 5 degrees
    convexity = edge_data.edge_convexity(angle_tol_rads)
    return convexity
#----------------------------------------------------------
#----------------------------------------------------------
def convexity_attribute(convexity, attribute):
    if attribute == "Convex edge":
        return convexity == EdgeConvexity.CONVEX
    if attribute == "Concave edge":
        return convexity == EdgeConvexity.CONCAVE
    if attribute == "Smooth":
        return convexity == EdgeConvexity.SMOOTH
    assert False, "Unknown convexity"
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def edge_length_attribute(edge):
    geometry_properties = GProp_GProps()
    brepgprop_LinearProperties(edge, geometry_properties)
    return geometry_properties.Mass()
#----------------------------------------------------------
#----------------------------------------------------------
def circular_edge_attribute(edge):
    brep_adaptor_curve = BRepAdaptor_Curve(edge)
    curv_type = brep_adaptor_curve.GetType()
    if curv_type == GeomAbs_Circle:
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def closed_edge_attribute(edge):
    if BRep_Tool().IsClosed(edge):
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def elliptical_edge_attribute(edge):
    brep_adaptor_curve = BRepAdaptor_Curve(edge)
    curv_type = brep_adaptor_curve.GetType()
    if curv_type == GeomAbs_Ellipse:
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def straight_edge_attribute(edge):
    brep_adaptor_curve = BRepAdaptor_Curve(edge)
    curv_type = brep_adaptor_curve.GetType()
    if curv_type == GeomAbs_Line:
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def hyperbolic_edge_attribute(edge):
    if Edge(edge).curve_type() == "hyperbola":
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def parabolic_edge_attribute(edge):
    if Edge(edge).curve_type() == "parabola":
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def bezier_edge_attribute(edge):
    if Edge(edge).curve_type() == "bezier":
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def non_rational_bspline_edge_attribute(edge):
    occwl_edge = Edge(edge)
    if occwl_edge.curve_type() == "bspline" and not occwl_edge.rational():
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def rational_bspline_edge_attribute(edge):
    occwl_edge = Edge(edge)
    if occwl_edge.curve_type() == "bspline" and occwl_edge.rational():
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def offset_edge_attribute(edge):
    if Edge(edge).curve_type() == "offset":
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def other_edge_attribute(edge):
    if Edge(edge).curve_type() == "other":
        return 1.0
    return 0.0
#----------------------------------------------------------
#----------------------------------------------------------
def extract_edge_attributes( edge_occ,faces_of_edge,attribute_config) -> list:
    
    # get the faces from an edge
    # top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
    # faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]
    #faces_of_edge = [f for f in self.body.faces_from_edge(Edge(edge_occ))]
    
    attribute_list = attribute_config["edge_attributes"]
    if "Concave edge" in attribute_list or \
            "Convex edge" in attribute_list or \
            "Smooth" in attribute_list:
        convexity = find_edge_convexity(edge_occ, faces_of_edge)
    #
    edge_attributes = []           #dim  [num_edges, 15]
    for attribute in attribute_list:
        if attribute == "Concave edge":
            edge_attributes.append(convexity_attribute(convexity, attribute))
        elif attribute == "Convex edge":
            edge_attributes.append(convexity_attribute(convexity, attribute))
        elif attribute == "Smooth":
            edge_attributes.append(convexity_attribute(convexity, attribute))
        elif attribute == "EdgeLengthAttribute":
            edge_attributes.append(edge_length_attribute(edge_occ))
        elif attribute == "CircularEdgeAttribute":
            edge_attributes.append(circular_edge_attribute(edge_occ))
        elif attribute == "ClosedEdgeAttribute":
            edge_attributes.append(closed_edge_attribute(edge_occ))
        elif attribute == "EllipticalEdgeAttribute":
            edge_attributes.append(elliptical_edge_attribute(edge_occ))
        elif attribute == "StraightEdgeAttribute":
            edge_attributes.append(straight_edge_attribute(edge_occ))
        elif attribute == "HyperbolicEdgeAttribute":
            edge_attributes.append(hyperbolic_edge_attribute(edge_occ))
        elif attribute == "ParabolicEdgeAttribute":
            edge_attributes.append(parabolic_edge_attribute(edge_occ))
        elif attribute == "BezierEdgeAttribute":
            edge_attributes.append(bezier_edge_attribute(edge_occ))
        elif attribute == "NonRationalBSplineEdgeAttribute":
            edge_attributes.append(non_rational_bspline_edge_attribute(edge_occ))
        elif attribute == "RationalBSplineEdgeAttribute":
            edge_attributes.append(rational_bspline_edge_attribute(edge_occ))
        elif attribute == "OffsetEdgeAttribute":
            edge_attributes.append(offset_edge_attribute(edge_occ))
        elif attribute == "Other":
            edge_attributes.append(other_edge_attribute(edge_occ))
        else:
            assert False, "Unknown face attribute"
    return edge_attributes          #dim  [num_edges, 15]
#----------------------------------------------------------
#----------------------------------------------------------
def extract_edge_grid( edge,faces_of_edge,num_crv_u) -> np.array:
    """
    Extract a edge grid (aligned with the coedge direction).
    
    The edge grids will be of size
    
        [ 12 x num_u ]
    
    The values are
        
        - x, y, z    (coordinates of the points)
        - tx, ty, tz (tangent of the curve, oriented to match the coedge)
        - Lx, Ly, Lz (Normal for the left face)
        - Rx, Ry, Rz (Normal for the right face)
    """
    # return dim [12 , num_crv_u]
    # get the faces from an edge
    # top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
    # faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]
    #faces_of_edge = [f for f in self.body.faces_from_edge(edge)]
    #
    edge_data = EdgeDataExtractor(edge, faces_of_edge, num_samples=num_crv_u, use_arclength_params=True)
    if not edge_data.good:
        # We hit a problem evaluating the edge data.  This may happen if we have
        # an edge with not geometry (like the pole of a sphere).
        # In this case we return zeros
        return np.zeros((12, num_crv_u)) #dim [12 , num_crv_u]
    
    single_grid = np.concatenate(
        [
            edge_data.points,
            edge_data.tangents,
            edge_data.left_normals,
            edge_data.right_normals
        ],
        axis=1
    )
    #
    return np.transpose(single_grid, (1, 0)) #dim [12 , num_crv_u]
#----------------------------------------------------------
#----------------------------------------------------------
def process(mapper,step_file,use_uv,attribute_config):
    '''
    通过mapper获取零件曲面属性、曲线属性、曲面拓扑邻接图；
    用于后续数据计算；
    '''
    # UV-gird size for face
    num_srf_u = attribute_config["UV-grid"]["num_srf_u"]
    num_srf_v = attribute_config["UV-grid"]["num_srf_v"]
    # UV-gird size for curve
    num_crv_u = attribute_config["UV-grid"]["num_crv_u"]
    #shape = load_body_from_step(step_file)
    #print('type(shape):',type(shape))
    ##
    #print("Begin: EntityMapper()")
    #mapper = EntityMapper(shape)
    #print("Done : EntityMapper()")
    #
    #assert shape is not None, "the shape {} is non-manifold or open".format(step_file)
    #assert self.checker(self.body.topods_shape()), "the shape {} has wrong topology".format(self.step_file)
    # 缩放零件几何到单位体大小
    #--------------------------
    #if self.scale_body:
    #    self.body = self.body.scale_to_unit_box(copy=True)
    #    # print(self.body.volume())
    #-----------------------------
    #
    try:
        graph = face_adjacency(mapper)
        print("graph: done")
        print('***time log***:',time.asctime())
    except Exception as e:
        print(e)
        assert False, 'Wrong shape {} when create face adjacency'.format(step_file)
    #
    # get the attributes for faces
    graph_face_attr = []
    graph_face_grid = [] # dim:(num_face_nodes, 7 , num_u , num_v )
    # the FaceCentroidAttribute has (x,y,z) coordinate , it`s 3 numbers
    # so the length of face attributes should add 2 if containing centroid
    len_of_face_attr = len(attribute_config["face_attributes"]) + 2 if "FaceCentroidAttribute" in attribute_config["face_attributes"] else 0
    for face_idx in graph.nodes:
        # Get the B-rep face
        face = Face(graph.nodes[face_idx]["face"])
        face_occ = face.topods_shape()
        #
        # type(face.surface()) -> OCC.Core.Geom.Geom_Surface
        if type(face.surface()) is float:
            continue
        #
        # get the attributes from face
        face_attr = extract_face_attributes(face_occ,attribute_config)  # from occwl.Face to OCC.TopoDS_Face
        assert len_of_face_attr == len(face_attr)
        graph_face_attr.append(face_attr)
        # get the UV point grid from face
        if use_uv and num_srf_u and num_srf_v:
            uv_grid = extract_face_grid( face,num_srf_u,num_srf_v)
            assert uv_grid.shape[0] == 7    # dim:(7 , num_u , num_v )
            graph_face_grid.append(uv_grid.tolist()) #torch.tensor 矩阵转换为 list格式 # dim:(7 , num_u , num_v )
    #
    print("graph_face_attr: done")
    print('***time log***:',time.asctime())
    graph_edge_attr = []
    graph_edge_grid = [] # dim:(num_edges,12 , num_u)
    for edge_idx in graph.edges:
        #edge_idx 的形式如下 [(0, 1),(0, 0),(0, 2),(0, 3)]
        edge = Edge(graph.edges[edge_idx]["edge"])
        edge_occ = edge.topods_shape()
        faces_of_edge =[]
        #for face_id in (list(graph.edges))[edge_idx]:
        edge_list = list(graph.edges(data=True)) 
        for face_id in edge_idx:
            faces_of_edge.append(Face(mapper.face_index_map[face_id]))
        # Ignore dgenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # get the attributes from edge
        edge_attr = extract_edge_attributes( edge_occ,faces_of_edge,attribute_config)
        assert len(attribute_config["edge_attributes"]) == len(edge_attr)
        graph_edge_attr.append(edge_attr)
        # get the UV point grid from edge
        if use_uv and num_crv_u:
            u_grid = extract_edge_grid( edge,faces_of_edge,num_crv_u)
            assert u_grid.shape[0] == 12   # dim:(12 , num_u)
            graph_edge_grid.append(u_grid.tolist())
    #
    print("graph_edge_attr: done")
    print('***time log***:',time.asctime())
    # get graph from nx.DiGraph
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    """
    #***--- 注意：这里又重构了graph  ---***
    """
    simple_graph = {
        'edges': (src, dst),
        'num_nodes': len(graph.nodes)
    }
    #
    graph_data = {
        'graph': graph,
        'simple_graph':simple_graph,
        'graph_face_attr': graph_face_attr,
        'graph_face_grid': graph_face_grid, # dim:(num_face_nodes, 7 , num_u , num_v )
        'graph_edge_attr': graph_edge_attr,
        'graph_edge_grid': graph_edge_grid, # dim:(num_edges,12 , num_u)
    }
    return graph_data
#
#
#------------------------------------------------------
#------------------------------------------------------
def get_angle_matrix(graph_data):
    node_feature = torch.tensor(graph_data['graph_face_grid'],dtype=torch.float32).permute(0,2,3,1)
    num_nodes = node_feature.shape[0]
    mean_normals_per_node = []
    #
    for i in range(num_nodes):
        normals = node_feature[i, :, :, 3:6]
        hidden_status = node_feature[i, :, :, 6]
        mask = (hidden_status == 0)
        
        if torch.any(mask):
            mask_expanded = mask.unsqueeze(-1).expand_as(normals)
            filtered_normals = normals[mask_expanded].view(-1, 3)
            mean_normal = torch.mean(filtered_normals, dim=0)
        else:
            mean_normal = torch.zeros(3)
        
        mean_normals_per_node.append(mean_normal)
    
    mean_normals_per_node = torch.stack(mean_normals_per_node)
    num_nodes = mean_normals_per_node.shape[0]
    angle_matrix = torch.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            dot_product = torch.dot(mean_normals_per_node[i], mean_normals_per_node[j])
            norm_i = torch.norm(mean_normals_per_node[i])
            norm_j = torch.norm(mean_normals_per_node[j])
            cos_theta = dot_product / (norm_i * norm_j)
            cos_theta = max(min(cos_theta, 1.0), -1.0)
            angle_radians = math.acos(cos_theta)
            angle_degrees = math.degrees(angle_radians)
            angle_matrix[i, j] = angle_degrees
            angle_matrix[j, i] = angle_degrees
    
    return angle_matrix
#------------------------------------------------------
#------------------------------------------------------
def get_centroid_distance_matrix(graph_data):
    # 获得曲面重心坐标相互距离矩阵；
    graph_face_attr = torch.tensor(graph_data["graph_face_attr"], dtype=torch.float32) #[num_nodes,feature])
    #
    node_centroid = graph_face_attr[: , 11:14]  # coordinates of center point
    
    node_centroid_matrix = node_centroid # face_centroid[num_nodes, 3(x,y,z)]
    #
    num_nodes = node_centroid_matrix.shape[0]
    expanded_a = node_centroid_matrix.unsqueeze(1).expand(-1, num_nodes, -1)
    expanded_b = node_centroid_matrix.unsqueeze(0).expand(num_nodes, -1, -1)
    distances = torch.norm(expanded_a - expanded_b, dim=2)
    #
    return distances
#------------------------------------------------------
#------------------------------------------------------
def get_shortest_distance_matrix(graph_data):
    #把图连接关系转换成曲面连接距离矩阵，然后找到里面最远连接距离；
    num_nodes = graph_data['simple_graph']['num_nodes']
    edges = torch.tensor(graph_data['simple_graph']['edges'])
    #
    graph = Data(edge_index=edges,num_nodes=num_nodes)
    G = to_networkx(graph,edge_attrs=None) 
    #
    #获取距离数据，形式为字典；
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    n = G.number_of_nodes()
    #把字典数据，转化为距离矩阵；
    distance_matrix = [[lengths[i].get(j, float('inf')) for j in range(n)] for i in range(n)]
    #
    max_distance = -np.inf
    # 遍历距离矩阵
    for i in range(n):
        for j in range(n):
            if distance_matrix[i][j] > max_distance and distance_matrix[i][j] != float('inf'):
                #
                max_distance = distance_matrix[i][j]
                #
    distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
    return distance_matrix , max_distance
#------------------------------------------------------
#------------------------------------------------------
def AddGlobalfeature(graph_data):
    # 根据抽取的几何数据，挖掘其他相关数据
    graph=graph_data['simple_graph']
    num_nodes = graph_data['simple_graph']['num_nodes']
    # 计算曲面间角度矩阵
    angle_matrix = get_angle_matrix(graph_data)
    # 获得曲面重心距离矩阵
    centroid_distance_matrix = get_centroid_distance_matrix(graph_data)
    # 获得节点距离矩阵
    shortest_distance_matrix , max_distance = get_shortest_distance_matrix(graph_data)
    #
    #更新字典
    graph_data['angle_matrix']=angle_matrix
    graph_data['centroid_distance_matrix']=centroid_distance_matrix
    graph_data['shortest_distance_matrix']=shortest_distance_matrix
    graph_data['edge_path_matrix']=torch.zeros(num_nodes, num_nodes, max_distance)
    #
    return graph_data
#------------------------------------------------------
#------------------------------------------------------
def read_stp(file_path):
    reader = STEPControl_Reader()
    reader.ReadFile(file_path)
    status = reader.TransferRoots()
    if status == IFSelect_RetDone:
        return reader.Shape()
    else:
        raise Exception("Failed to read STEP file")
#------------------------------------------------------
#------------------------------------------------------
def get_sample_list(folder_path,attribute_config):
    #获取数据集路径内的step零件样本列表；
    #假定数据集命名规则：GFR_00001.step(样本); GFR_00001_rib.step（特征）;
    sample_list = [f for f in os.listdir(folder_path) if f.lower().endswith('.step') and 'base' not in f]
    record_list = []
    for f in sample_list:
        for key in attribute_config['lables'].keys():  #tmp_key=f.lower().replace('.step','').split('_')[-1]
            if key.lower() in f.lower():
                record_list.append(f)
    for f in record_list:
        sample_list.remove(f)
    #print('sample_list:',sample_list)
    return sample_list
#------------------------------------------------------
#------------------------------------------------------
def dist_2_Pnt(a,b):
    powe=math.pow
    sqrt = math.sqrt
    #用于计算两个空间点的笛卡尔距离
    # a=[x1,y1,z1] ;b=[x2,y2,z2]
    return( sqrt(powe((a[0]-b[0]),2)+powe((a[1]-b[1]),2)+powe((a[2]-b[2]),2)))
#------------------------------------------------------
#------------------------------------------------------
def get_sameface_list(shape_base,shape_feature):
    #获取base_part中，两个零件的相同面的面号列表；
    geom_Prop = GProp_GProps()
    brep_SP = brepgprop_SurfaceProperties
    #
    faces_base= list(shape_base.faces())
    faces_feature = list(shape_feature.faces())
    #
    base_data = [(brep_SP(a1.topods_shape(), geom_Prop),geom_Prop.Mass(),geom_Prop.CentreOfMass().Coord()) for a1 in faces_base]
    feature_data = [(brep_SP(a1.topods_shape(), geom_Prop),geom_Prop.Mass(),geom_Prop.CentreOfMass().Coord()) for a1 in faces_feature]
    #
    count = 0
    base_list = []
    feature_list = []
    #
    for i in range(len(base_data)):
        if i not in base_list:
            for j in range(len(feature_data)):
                if j not in feature_list:
                    if abs(base_data[i][1]-feature_data[j][1])<1e-1:
                        if dist_2_Pnt(base_data[i][2],feature_data[j][2])<1e-1:
                            base_list.append(i)
                            feature_list.append(j)
                            count +=1
                            #print(i,j)
                            continue
    #print('count:',count)
    return(base_list)
#------------------------------------------------------
#------------------------------------------------------
class BrepFeaturePointAnalyzer:
    def __init__(self, tolerance=1e-6):
        """
        初始化BRep特征点分析器
        
        参数:
            tolerance: 点匹配容差
        """
        self.tolerance = tolerance
        self.point_id_to_coord = {}  # 点ID到坐标的映射
        self.point_id_to_faces = defaultdict(set)  # 点ID到拥有该点的面的映射
        self.face_id_to_points = defaultdict(set)  # 面ID到该面包含的点ID的映射
        self.face_id_to_actual_face = {}  # 面ID到实际TopoDS_Face对象的映射
        self.next_face_id = 0  # 面ID计数器
        #
        
    def _quantize_point(self, p: gp_Pnt) -> str:
        """
        将点坐标量化为容差网格并生成唯一ID
        
        参数:
            p: 要量化的点
            
        返回:
            点的唯一ID字符串
        """
        x = p.X()
        y = p.Y()
        z = p.Z()
        return (f"{x:.1f}_{y:.1f}_{z:.1f}")
    
    def extract_face_feature_points(self, face: TopoDS_Face) -> list:
        """
        从单个面提取所有特征点
        
        参数:
            face: 要分析的面对象
            
        返回:
            该面的特征点ID列表
        """
        points = []
        
        ## 提取曲面控制点
        #surface = BRep_Tool.Surface(face)
        #if surface and surface.IsKind("Geom_BSplineSurface"):
        #    bspline = surface.GetObject()
        #    for i in range(1, bspline.NbUPoles() + 1):
        #        for j in range(1, bspline.NbVPoles() + 1):
        #            pole = bspline.Pole(i, j)
        #            points.append(pole)
        
        # 提取边界顶点
        explorer = TopExp_Explorer(face, TopAbs_VERTEX)
        while explorer.More():
            vertex = explorer.Current()
            pnt = BRep_Tool.Pnt(vertex)
            points.append(pnt)
            explorer.Next()
            
        return points
    def extract_shape_feature_points(self, shape: TopoDS_Shape) -> list:
        """
        从单个面提取所有特征点
        
        参数:
            face: 要分析的面对象
            
        返回:
            该面的特征点ID列表
        """
        points = []
        
        # 提取边界顶点
        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        while explorer.More():
            vertex = explorer.Current()
            pnt = BRep_Tool.Pnt(vertex)
            points.append(pnt)
            explorer.Next()
            
        return points
    
    def load_brep_from_step(self, file_path: str):
        """
        从STEP文件加载BRep模型并分析所有面的特征点
        
        参数:
            file_path: STEP文件路径
        """
        # 加载STEP文件
        reader = STEPControl_Reader()
        reader.ReadFile(file_path)
        reader.TransferRoot()
        shape = reader.Shape()
        
        #获取体vertex
        points = self.extract_shape_feature_points(shape) 
        #
        for p in points:
                pid = self._quantize_point(p)
                
                # 记录点坐标(如果是新点)
                if pid not in self.point_id_to_coord:
                    self.point_id_to_coord[pid] = p

        # 遍历所有面
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods_Face(explorer.Current())
            face_id = self.next_face_id
            self.next_face_id += 1
            self.face_id_to_actual_face[face_id] = face
            
            # 提取特征点
            points = self.extract_face_feature_points(face)
            for p in points:
                pid = self._quantize_point(p)
                
                # 记录点与面的双向关系
                self.point_id_to_faces[pid].add(face_id)
                self.face_id_to_points[face_id].add(pid)
                
            explorer.Next()
    
    def find_shared_points_with(self, other_analyzer):
        """
        查找两个分析器共有的特征点
        
        参数:
            other_analyzer: 另一个BrepFeaturePointAnalyzer实例
            
        返回:
            共有的点ID集合
        """
        print('本模型点数：',len(self.point_id_to_coord.keys()))
        print('对比模型点数：',len(other_analyzer.point_id_to_coord.keys()))

        print('本模型面数：',len(self.face_id_to_points.keys()))
        print('对比模型面数：',len(other_analyzer.face_id_to_points.keys()))

        return set(self.point_id_to_coord.keys()) & set(
            other_analyzer.point_id_to_coord.keys()
        )
    
    def get_face_ids_by_point_id(self, point_id: str) -> set:
        """
        根据点ID获取拥有该点的面ID集合
        
        参数:
            point_id: 点的唯一ID
            
        返回:
            包含该点的面ID集合
        """
        return self.point_id_to_faces.get(point_id, set())
    
    def get_point_coord_by_id(self, point_id: str) -> gp_Pnt:
        """
        根据点ID获取实际坐标
        
        参数:
            point_id: 点的唯一ID
            
        返回:
            点的gp_Pnt对象
        """
        return self.point_id_to_coord.get(point_id)
    
    def get_face_by_id(self, face_id: int) -> TopoDS_Face:
        """
        根据面ID获取实际的面对象
        
        参数:
            face_id: 面的唯一ID
            
        返回:
            TopoDS_Face对象
        """
        return self.face_id_to_actual_face.get(face_id)
    
    def get_points_by_face_id(self, face_id: int) -> set:
        """
        根据面ID获取该面包含的所有点ID
        
        参数:
            face_id: 面的唯一ID
            
        返回:
            该面包含的点ID集合
        """
        return self.face_id_to_points.get(face_id, set())
#
#------------------------------------------------------
#------------------------------------------------------
#
def find_matching_faces(analyzer1: BrepFeaturePointAnalyzer, 
                        analyzer2: BrepFeaturePointAnalyzer) -> list:
    """
    查找两个BRep模型中相同的面
    
    参数:
        analyzer1: 第一个BRep模型的分析器
        analyzer2: 第二个BRep模型的分析器
        
    返回:
        匹配的面ID对列表 [(analyzer1_face_id, analyzer2_face_id), ...]
    """
    matching_faces = []
    
    # 步骤1: 找出共有的特征点
    shared_points = analyzer1.find_shared_points_with(analyzer2)
    print('匹配的点数:',len(shared_points))
    
    # 步骤2: 对每个共享点找出它所属的两个模型的面
    point_to_face_mapping = defaultdict(lambda: (set(), set()))
    for pid in shared_points:
        analyzer1_faces = analyzer1.get_face_ids_by_point_id(pid)
        analyzer2_faces = analyzer2.get_face_ids_by_point_id(pid)
        point_to_face_mapping[pid] = (analyzer1_faces, analyzer2_faces)
    
    # 步骤3: 找出可能匹配的面(共享至少一个点的面组合)
    candidate_pairs = set()
    for pid, (faces1, faces2) in point_to_face_mapping.items():
        for f1 in faces1:
            for f2 in faces2:
                candidate_pairs.add((f1, f2))
    
    # 步骤4: 验证候选面组合的点集是否完全相同
    for f1, f2 in candidate_pairs:
        points1 = analyzer1.get_points_by_face_id(f1)
        points2 = analyzer2.get_points_by_face_id(f2)
        
        # 点集完全相同才认为面相同
        if points1 == points2:
            if f1 not in matching_faces:
                matching_faces.append(f1)
    print("匹配的面数:",len(matching_faces))
    return matching_faces
#
#------------------------------------------------------
#------------------------------------------------------
#
def get_graph_label(mapper,folder_path,file_name, attribute_config,graph_data):
    #获取file_name对应的step文件的特征标签信息；特征标签序号存储于attribute_config.json文件
    label_dict = {}
    file_base = os.path.join(folder_path,file_name)
    analyzer_base = BrepFeaturePointAnalyzer()
    #
    analyzer_base.load_brep_from_step(file_base)
    #
    num_faces = mapper.get_num_faces()
    for face_idx in range(num_faces):
        label_dict[str(face_idx)]=0
    #
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith('.step')]
    for f in file_list:
        if file_name[:-5].lower() in f.lower():
            #tmp_key=os.path.split(f)[1].lower().strip('.step').split('_')[-1]
            tmp_key = Path(f).stem.lower().split('_')[-1]
            if tmp_key in attribute_config['lables'].keys():
                #
                print('find feature:',tmp_key)
                file_feature = os.path.join(folder_path,f)
                analyzer_feature = BrepFeaturePointAnalyzer()
                analyzer_feature.load_brep_from_step(file_feature)
                #查找匹配的面
                matching_faces = find_matching_faces(analyzer_base, analyzer_feature)
                #
                # 更新特征编号
                for face_idx in matching_faces:
                    label_dict[str(face_idx)]=attribute_config['lables'][tmp_key]
    #            
    graph_data['label_feature']=label_dict
    #假定数据集样本命名规则：GFR_00001.step, GFR_00002.step, GFR_00003.step #输出id为：1,2,3...#
    graph_data['id']=torch.tensor([int(file_name.lower().replace('.step','').split('_')[-1])]).to(dtype = torch.long)
    #
    #
    return(graph_data)
#
#------------------------------------------------------
#------------------------------------------------------
def pre_save_graph_data(graph_data):
    #用于 graph_data 数据化存储，去除networkx生成的graph图，保留simple_graph代替
    ''' 数据示意：
    graph_data = {
        'graph'                     : graph,
        'simple_graph'              :simple_graph,
        'graph_face_attr'           : graph_face_attr,
        'graph_face_grid'           : graph_face_grid,
        'graph_edge_attr'           : graph_edge_attr,
        'graph_edge_grid'           : graph_edge_grid,
        'angle_matrix'              : angle_matrix,
        'centroid_distance_matrix'  : centroid_distance_matrix,
        'shortest_distance_matrix'  : shortest_distance_matrix,
        'edge_path_matrix'          :torch.zeros(num_nodes, num_nodes, max_distance)
        'label_feature'             : label_dict,
        'id'                        : id
    }
    '''
    #
    new_graph_data={}
    #临接图数据
    new_graph_data['simple_graph']=graph_data['simple_graph']
    #面节点数据
    #new_graph_data['node_datas']=torch.tensor(graph_data["graph_face_grid"], dtype=torch.float32).permute(0, 2, 3, 1)
    new_graph_data['node_datas']=torch.tensor(graph_data["graph_face_grid"], dtype=torch.float32)
    graph_face_attr = torch.tensor(graph_data["graph_face_attr"], dtype=torch.float32)  # [num_nodes,feature])
    new_graph_data['node_area']     = graph_face_attr[:, 9]
    new_graph_data['node_type']     = graph_face_attr[:, 0:9]
    new_graph_data['node_centroid'] = graph_face_attr[:, 11:14]
    new_graph_data['node_rational'] = torch.tensor(graph_face_attr[:, 10], dtype=torch.long) 
    #边数据
    new_graph_data['edge_datas']  =torch.tensor(graph_data["graph_edge_grid"], dtype=torch.float32).permute(0, 2, 1)
    graph_edge_attr = torch.tensor(graph_data["graph_edge_attr"], dtype=torch.float32)
    new_graph_data['edge_type']      = graph_edge_attr[:, 4:15]
    new_graph_data['edge_convexity'] = graph_edge_attr[:, 0:3]
    new_graph_data['edge_length']    = graph_edge_attr[:, 3]
    #基于元数据的计算所得中间数据，数据量交大，未来考虑放在计算过程中生成
    #转移到函数 graph_data_enrich_in_calculation() 
    #graph=graph_data['graph']
    #n_nodes = graph_data['simple_graph']['num_nodes']
    #dense_adj = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.long)
    #node_degree = dense_adj.long().sum(dim=1).view(-1)
    #attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float)
    #new_graph_data['node_degree']        = node_degree
    #new_graph_data['attn_biases']        = attn_bias
    #面节点其他关系
    #new_graph_data['angles']             = graph_data['angle_matrix']
    #new_graph_data['centroid_distances'] = graph_data['centroid_distance_matrix']
    #new_graph_data['shortest_distances'] = graph_data['shortest_distance_matrix']
    #new_graph_data['edge_paths']         = graph_data['edge_path_matrix']
    #
    new_graph_data['face_labels']=graph_data['label_feature']
    new_graph_data['data_ids']    =graph_data['id']
    #添加 'node_degree' 'attn_biases'
    n_nodes = graph_data['simple_graph']['num_nodes']
    matrix = torch.tensor(graph_data['simple_graph']['edges'],dtype=torch.long)
    matrix = matrix.t()
    graph = nx.DiGraph()
    graph.add_edges_from(matrix)
    dense_adj = torch.tensor(nx.adjacency_matrix(graph).todense(),dtype=torch.long)
    node_degree = dense_adj.long().sum(dim=1).view(-1)
    #
    #attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float32)
    #
    new_graph_data['node_degree']        = node_degree
    #new_graph_data['attn_biases']        = attn_bias
    #
    return(new_graph_data)
#------------------------------------------------------
#------------------------------------------------------
def graph_data_enrich_in_calculation(graph_data):
    matrix = torch.tensor(graph_data['simple_graph']['edges'],dtype=torch.long)
    matrix = matrix.t()
    graph = nx.DiGraph()
    graph.add_edges_from(matrix)
    dense_adj = torch.tensor(nx.adjacency_matrix(graph).todense(),dtype=torch.long)
    node_degree = dense_adj.long().sum(dim=1).view(-1)
    #
    attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float32)
    #
    graph_data['node_degree']        = node_degree
    graph_data['attn_biases']        = attn_bias
    return(graph_data)
#------------------------------------------------------
#------------------------------------------------------
#def process_one_file(args,output_path,file_name,folder_path,graph_data):
def process_one_file(file_name,output_path,folder_path,attribute_config):
    #
    print("#####-----Processing---------#####:",file_name)
    print('***time log***:',time.asctime())
    # UV-gird size for face
    num_srf_u = attribute_config["UV-grid"]["num_srf_u"]
    num_srf_v = attribute_config["UV-grid"]["num_srf_v"]
    # UV-gird size for curve
    num_crv_u = attribute_config["UV-grid"]["num_crv_u"]
    #file_path, attribute_config = args
    # 输入几何文件路径
    input_file_path = os.path.join(folder_path, file_name)
    # 构建输出文件的完整路径
    output_file_path = os.path.join(output_path, file_name.replace('.step','.pkl'))
    #
    # 检查输出路径下是否已存在同名.bin文件
    if os.path.exists(output_file_path):
        print(f"File {output_file_path} already exists, skipping...")
        return [str(Path(input_file_path).stem)]
    else:
        #
        try:
            shape = load_body_from_step(input_file_path)
            print('step file readin:done',file_name)
            #
            #print('***time log***:',time.asctime())
            print("Begin: EntityMapper()",file_name)
            mapper = EntityMapper(shape)
            print("Done : EntityMapper()",file_name)
            #print('***time log***:',time.asctime())
            #
            print('Begin:process()',file_name)
            graph_data = process(mapper,input_file_path,use_uv,attribute_config)
            print('Done:process()',file_name)
            #print('***time log***:',time.asctime())
            #
            #print('Begin:AddGlobalfeature')
            #graph_data = AddGlobalfeature(graph_data)
            #print('Done:AddGlobalfeature')
            #print('***time log***:',time.asctime())
            #
            print('Begin:get_graph_label',file_name)
            graph_data = get_graph_label(mapper,folder_path,file_name, attribute_config,graph_data)
            print('Done:get_graph_label',file_name)
           # print('***time log***:',time.asctime())
            #数据存储前预处理
            print('Begin:pre_save_graph_data',file_name)
            graph_data = pre_save_graph_data(graph_data)
            print('Done:pre_save_graph_data',file_name)
            #print('***time log***:',time.asctime())
            #
            ##以下操作后，可以把数据输入到网络BrepSeg
            #graph_data=[graph_data]
            #print('Begin:collator')
            #graph_data = collator(graph_data,32,64)
            #print('Done:collator')
            #print('***time log***:',time.asctime())
            ##
            #
            print('Begin:save_graph_data',file_name)
            #把图数据写入文件保存
            with open(Path(output_file_path) , 'wb') as f:
                pickle.dump(graph_data,f)
            #
            print('Done:save_graph_data',file_name)
            #print('***time log***:',time.asctime())
            return [str(Path(input_file_path).stem)]
        except Exception as e:
            print('\n*--*--ERROR*--*--:\n',e,file_name)
            return []
#------------------------------------------------------
#------------------------------------------------------
def pad_mask_unsqueeze(x, padlen):  
    # 输入：x维度为1维向量，函数功能为用1补齐向量长度到padlen
    # 输出：维度为（1，padlen）
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_ones([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)
#------------------------------------------------------
#------------------------------------------------------
def pad_attn_bias_unsqueeze(x, padlen):  # （2）
    #输入：x为方阵（n,n）
    #输出：维度为（1，padlen，padlen）
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)
#------------------------------------------------------
#------------------------------------------------------
def pad_shortest_distance_unsqueeze(x, padlen):  # x[num_nodes, num_nodes]   （3）
    #输入：x为方阵（n,n）
    #输出：维度为（1，padlen，padlen）
    x = x + 1  # Prevent dividing by 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)
#------------------------------------------------------
#------------------------------------------------------
def pad_angle_unsqueeze(x, padlen):  # x[num_nodes, num_nodes]   （3）
    #输入：x为方阵（n,n）
    #输出：维度为（1，padlen，padlen）
    x = x + 1.0  # Prevent dividing by 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)
#------------------------------------------------------
#------------------------------------------------------
def pad_centroid_distance_unsqueeze(x, padlen):  # x[num_nodes, num_nodes]   （3）
    #输入：x为方阵（n,n）
    #输出：维度为（1，padlen，padlen）
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)
#------------------------------------------------------
#------------------------------------------------------
def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):  # x[num_nodes, num_nodes, max_dist]  （6）
    #输入：x为方阵（m,n,q）
    #输出：维度为（1，padlen1，padlen2,padlen3）
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = -1 * x.new_ones([padlen1, padlen2, padlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    return x.unsqueeze(0)
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
def nx_batch(graphs):
    #
    tmp_graphs = []
    for graph_data in graphs:
        matrix = torch.tensor(graph_data['edges'],dtype=torch.long)
        matrix = matrix.t()
        graph = nx.DiGraph()
        graph.add_edges_from(matrix)
        tmp_graphs.append(graph)
    #
    # 计算节点ID偏移量
    num_nodes = [G.number_of_nodes() for G in tmp_graphs]
    offsets = [0] + list(accumulate(num_nodes))[:-1]  # 偏移量列表
    
    # 创建新图
    batched_G = nx.DiGraph()
    #
    # 合并节点和边（重新编号）
    for i, (G, offset) in enumerate(zip(tmp_graphs, offsets)):
        # 添加节点（可选：添加属性标记子图ID）
        for node in G.nodes():
            batched_G.add_node(node + offset, subgraph_id=i)
        # 添加边
        for u, v in G.edges():
            batched_G.add_edge(u + offset, v + offset)
    # 保存子图信息
    batched_G.graph['batch_num_nodes'] = num_nodes
    batched_G.graph['batch_num_edges'] = [G.number_of_edges() for G in tmp_graphs]
    #
    return batched_G
#------------------------------------------------------
#------------------------------------------------------
def collator(items, multi_hop_max_dist,spatial_pos_max):  
    #items的数据形式为列表，列表[graph_data01,graph_data02,graph_data03,graph_data04...]
    """
    items = [
    (
        graph1.graph,
        graph1.node_data,
        graph1.face_area,
        # ... 其他属性
        graph1.edge_path[:, :, :multi_hop_max_dist],
        graph1.label_feature,
        graph1.data_id
    ),....
    ]
    """
    items = [
        (
            item['simple_graph'],
            item['node_datas'],  # node_data[num_nodes, U_grid, V_grid, pnt_feature]
            item['node_area'],  # face_area[num_nodes]
            item['node_type'],  # face_type[num_nodes,9(one-hot)]
            item['node_centroid'],  # face_centroid[num_nodes, 3(x,y,z)]
            item['node_rational'],  # face_rational[num_nodes]
            item['edge_datas'],  # edge_data[num_edges, U_grid, pnt_feature]
            item['edge_type'],  # edge_type[num_edges,11(one-hot)]
            item['edge_length'],  # edge_len[num_edges]
            item['edge_convexity'],  # edge_ang[num_edges,3(one-hot)]
            item['node_degree'],
            item['attn_biases'],  # attn_bias[num_nodes + 1, num_nodes + 1]
            item['angles'],
            item['centroid_distances'],
            item['shortest_distances'],
            item['edge_paths'][:, :, :multi_hop_max_dist],  # [num_nodes, num_nodes, max_dist]
            item['face_labels'],  # [num_nodes]
            item['data_ids']
        )
        for item in items
    ]
    #
    (
        graphs,
        node_datas, # node_datas[num_graphs, num_nodes, U_grid, V_grid, pnt_feature]
        face_areas,
        face_types,
        face_centroids,
        face_rationals,
        edge_datas,
        edge_types,
        edge_lengths,
        edge_convexitys,
        node_degrees,
        attn_biases,
        angles,
        centroid_distances,
        shortest_distances,
        edge_paths,
        label_features,
        data_ids
    ) = zip(*items)  # 解压缩
    #标记符合shortest_distances[idx] >= spatial_pos_max的数据点为-inf,目的让网络忽略该信息
    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][shortest_distances[idx] >= spatial_pos_max] = float("-inf")
    #--------------------------------------------------------------------
    max_node_num = max(i.size(0) for i in node_datas)  # 计算这批数据中图节点的最大数量
    max_edge_num = max(i.size(0) for i in edge_datas)  # 计算这批数据的最大边数

    max_dist = max(i.size(-1) for i in edge_paths)  # 计算节点间的最大距离 针对某些图的max_dist都小于multi_hop_max_dist的情况
    max_dist = max(max_dist, multi_hop_max_dist)

    # 对数据进行打包并返回, 将各数据调整到同一长度，以max_node_num为准

    # --------------------------------------------------------------------#
    # padding_mask_list:图长度掩码，以False填充 # dim[num_graphs, num_nodes]
    padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in node_datas] # dim[num_graphs, num_nodes]
    # node_padding_mask: 图统一长度掩码，矩阵用1补全原矩阵以外的维度 # dim[num_graphs, max_node_num]
    node_padding_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in padding_mask_list]) # dim[num_graphs, max_node_num]
    
    # edge_padding_mask_list：边长度掩码，以False填充
    edge_padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in edge_datas]
    # edge_padding_mask: 边统一长度掩码，矩阵用1补全原矩阵以外的维度
    edge_padding_mask = torch.cat([pad_mask_unsqueeze(i, max_edge_num) for i in edge_padding_mask_list])
    # --------------------------------------------------------------------#
    
    # 节点特征
    node_data = torch.cat([i for i in node_datas])  # node_datas=(batch_size, [num_nodes, U_grid, V_grid, pnt_feature])
    face_area = torch.cat([i for i in face_areas])  # [total_node_num]
    face_type = torch.cat([i for i in face_types])  # [total_node_num,9(one-hot)]
    face_centroid = torch.cat([i for i in face_centroids])  # [total_node_num,3(x,y,z)]
    face_rational = torch.cat([torch.tensor(i,dtype=torch.long) for i in face_rationals])  # [total_node_num]
    
    # 边特征
    edge_data = torch.cat([i for i in edge_datas])  # edge_datas(batch_size, [num_edges, U_grid, pnt_feature])元组
    edge_type = torch.cat([i for i in edge_types])  # []
    edge_length = torch.cat([i for i in edge_lengths])
    edge_convexity = torch.cat([i for i in edge_convexitys])
    
    # 边编码输入
    edge_path = torch.cat(  # edges_paths(batch_size, [num_nodes, num_nodes, max_dist])
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_paths]
    ).long()
    
    # 注意力矩阵
    attn_bias = torch.cat(  # attn_bias(batch_size, [num_nodes+1, num_nodes+1]) 多了一个全图的虚拟节点
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    
    # 空间编码
    shortest_distance = torch.cat(  # shortest_distances(batch_size, [num_nodes, num_nodes])
        [pad_shortest_distance_unsqueeze(i, max_node_num) for i in shortest_distances]
    )
    
    angle = torch.cat(
        [pad_angle_unsqueeze(i, max_node_num) for i in angles]
    )
    
    centroid_distance = torch.cat(
        [pad_centroid_distance_unsqueeze(i, max_node_num) for i in centroid_distances]
    )
    
    # 中心性编码
    in_degree = torch.cat([i for i in node_degrees])
    #
    '''
    graphs
    matrix = torch.tensor(graph_data['simple_graph']['edges'],dtype=torch.long)
    matrix = matrix.t()
    graph = nx.DiGraph()
    graph.add_edges_from(matrix)
    '''
    #
    batched_graph = nx_batch(graphs)
    
    # face feature type
    batched_label_feature = torch.cat([torch.tensor(list(i.values()),dtype=torch.long) for i in label_features])
    
    data_ids = torch.tensor([i for i in data_ids])
    
    batch_data = dict(
        
        node_padding_mask=node_padding_mask,  # [batch_size, max_node_num]
        edge_padding_mask=edge_padding_mask.type(torch.bool),  # [batch_size, max_edge_num]
        graph=batched_graph,

        node_data=node_data,  # [total_node_num, U_grid, V_grid, pnt_feature] # total是这个batch里所有的node数
        face_area=face_area,  # [total_node_num]
        face_type=face_type,  # [total_node_num,9(one-hot)]
        face_centroid=face_centroid,  # [total_node_num,3(x,y,z)]
        face_rational=face_rational,  # [total_node_num]

        edge_data=edge_data,  # [total_edge_num, U_grid, pnt_feature]
        edge_type=edge_type,  # [total_edge_num,11(one-hot)]
        edge_length=edge_length,  # [total_edge_num]
        edge_convexity=edge_convexity,  # [total_edge_num,3(one-hot)]

        in_degree=in_degree,  # [batch_size, max_node_num]
        out_degree=in_degree,  # [batch_size, max_node_num] #无向图
        attn_bias=attn_bias,  # [batch_size, max_node_num+1, max_node_num+1]

        shortest_distance=shortest_distance,  # [batch_size, max_node_num, max_node_num]
        angle=angle,  # [batch_size, max_node_num, max_node_num]
        centroid_distance=centroid_distance,  # [batch_size, max_node_num, max_node_num]
        edge_path=edge_path,  # [batch_size, max_node_num, max_node_num, max_dist] 空位用-1填充

        label_feature=batched_label_feature,  # [total_node_num]
        id=data_ids
    )
    return batch_data
#------------------------------------------------------
#------------------------------------------------------
def move_to_device(batch_data, device):
    for key in batch_data:
        value = batch_data[key]
        # 检查当前元素是否为张量
        if isinstance(value, torch.Tensor):
            # 如果是张量，则将其迁移到指定设备
            batch_data[key] = value.to(device)
        # 对于其他类型的数据，比如 DGL 图或列表等，可以进行额外处理
        elif hasattr(value, 'to'):
            # 例如，DGL图可能有一个 .to() 方法
            try:
                batch_data[key] = value.to(device)
            except Exception as e:
                print(f"Could not move {key} to device. Error: {e}")
        # 其他类型的值不做处理
    return batch_data
#------------------------------------------------------
#------------------------------------------------------
def paralell_data_processing():
    
    return()
#------------------------------------------------------
#------------------------------------------------------
def initializer():
    import signal
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
#------------------------------------------------------
#------------------------------------------------------
if __name__ == '__main__':
    #step_path = "/home/zhx/project/gfr/dataset/thin_parts"
    step_path = "/Dataset/GFR_Dataset"
    output_path    = "/Dataset/GFR_data_slim"
    attribute_config_path = "/home/zhx/project/yb_gfr/attribute_config.json"
    num_workers = 16
    #
    step_path = Path(step_path)
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir()
    attribute_config_path = Path(attribute_config_path)
    #
    attribute_config = load_json(attribute_config_path)
    #step_files = list(step_path.glob("*.st*p"))
    step_files = get_sample_list(step_path,attribute_config)
    print('step_files:',step_files)
    #**********************************************************
    scale_body = True
    # whether to extract UV-grid
    checker = TopologyChecker()
    # UV-gird size for face
    num_srf_u = attribute_config["UV-grid"]["num_srf_u"]
    num_srf_v = attribute_config["UV-grid"]["num_srf_v"]
    # UV-gird size for curve
    num_crv_u = attribute_config["UV-grid"]["num_crv_u"]
    use_uv = True
    body = None
    #***********************************************************
    #for file_name in step_files:
    #    graph_data = process_one_file(file_name,output_path,step_path,attribute_config)
    
    
    # 创建一个partial函数，预设attribute_config参数
    partial_process_one_file = partial(process_one_file, output_path = output_path,folder_path=step_path,attribute_config=attribute_config)
    #
    pool = Pool(processes=num_workers, initializer=initializer,)
    try:
        results = list(tqdm(
            pool.imap(partial_process_one_file, step_files),total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    #
    pool.terminate()
    pool.join()
    #
    graph_count = 0
    fail_count = 0
    graphs = []
    for res in results:
        if len(res) > 0:
            graph_count += 1
            graphs.append(res)
        else:
            fail_count += 1
    #
    gc.collect()
    print(f"Process {len(results)} files. Generate {graph_count} graphs. Has {fail_count} failed files.")
    
    
    #graph_data = collator(graph_data,32,64)
    #device='cuda'
    #graph_data = move_to_device(graph_data02, device)
#------------------------------------------------------
#------------------------------------------------------