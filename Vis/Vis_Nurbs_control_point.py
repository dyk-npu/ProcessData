# -*- coding: utf-8 -*-

"""
B-rep NURBS曲面控制点可视化脚本 (基于 occwl.viewer 和 PyQt) - v3 可选面版

功能:
    加载一个STEP文件，并可选择性地只对指定的几个面进行控制点和
    控制多边形的可视化。其余部分将作为灰色背景显示。
"""

import os
import sys
import argparse
import numpy as np

# 核心 OCC 和 occwl 依赖
from occwl.viewer import Viewer
from occwl.compound import Compound
from occwl.edge import Edge
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge


def array_from_Array2OfPnt(array):
    """将OCC的Geom_Pnt二维数组高效转换为Numpy数组。"""
    return np.asarray(
        [
            [array.Value(i, j).Coord() for j in range(array.LowerCol(), array.UpperCol() + 1)]
            for i in range(array.LowerRow(), array.UpperRow() + 1)
        ]
    )

def create_edge_from_points(p1, p2):
    """使用 BRepBuilderAPI_MakeEdge 从两个Numpy点创建 occwl.edge.Edge 对象。"""
    gp_p1 = gp_Pnt(p1[0], p1[1], p1[2])
    gp_p2 = gp_Pnt(p2[0], p2[1], p2[2])
    builder = BRepBuilderAPI_MakeEdge(gp_p1, gp_p2)
    if builder.IsDone():
        return Edge(builder.Edge())
    return None


def visualize_control_points(viewer, shape, 
                             indices_to_show=None,
                             show_points=True, 
                             show_hull=True,
                             point_color=(1, 0, 0),       # 红色
                             point_scale=5,
                             hull_color=(0, 0, 0),        # 黑色
                             highlight_color=(0.1, 0.6, 1.0), # 高亮的蓝色
                             default_color=(0.8, 0.8, 0.8), # 默认的灰色
                             transparency=0.7):
    """
    在给定的viewer中绘制几何实体指定面的控制点和控制多边形。

    Args:
        viewer (occwl.viewer.Viewer): 要在其上绘制的查看器实例。
        shape (occwl.solid.Solid or occwl.shell.Shell): 要处理的几何实体。
        indices_to_show (list of int, optional): 要高亮显示并分析的面索引列表。
                                                 如果为 None，则处理所有面。
        ... (其他可视化参数)
    """
    if not hasattr(shape, 'faces'):
        print("错误: 提供的对象没有 'faces' 属性。")
        return

    all_faces = list(shape.faces())
    
    faces_to_process = []
    if indices_to_show is None:
        # 如果没有指定索引，则处理所有面
        print(f"开始处理全部 {len(all_faces)} 个面...")
        faces_to_process = list(enumerate(all_faces))
    else:
        # 如果指定了索引，则只处理这些面，并先绘制背景
        print(f"为模型添加灰色背景...")
        viewer.display(shape, color=default_color, transparency=transparency)
        
        valid_indices = [i for i in indices_to_show if i < len(all_faces)]
        if len(valid_indices) != len(indices_to_show):
            print(f"警告: 部分提供的面索引超出了模型总面数({len(all_faces)})，已被忽略。")

        print(f"开始处理指定的 {len(valid_indices)} 个面: {valid_indices}")
        faces_to_process = [(i, all_faces[i]) for i in valid_indices]

    for i, face in faces_to_process:
        try:
            surface_handle = BRep_Tool.Surface(face.topods_shape())
            bspline_surface = geomconvert.SurfaceToBSplineSurface(surface_handle)
        except Exception:
            continue

        # 重新显示（或第一次显示）当前面，使用高亮颜色
        viewer.display(face, color=highlight_color, transparency=transparency - 0.2)
        
        poles_occ = bspline_surface.Poles()
        poles_np = array_from_Array2OfPnt(poles_occ)
        
        if show_points:
            points_flat = poles_np.reshape(-1, 3)
            viewer.display_points(points_flat, color=point_color, scale=point_scale)

        if show_hull:
            for row in poles_np:
                for j in range(len(row) - 1):
                    edge = create_edge_from_points(row[j], row[j+1])
                    if edge: viewer.display(edge, color=hull_color)
            
            for col in poles_np.T:
                for j in range(len(col) - 1):
                    edge = create_edge_from_points(col[j], col[j+1])
                    if edge: viewer.display(edge, color=hull_color)
    
    print("可视化处理完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("B-rep NURBS曲面控制点可视化脚本 (可选面版)")
    parser.add_argument("--input_file", type=str, help="要可视化的STEP文件路径")
    args = parser.parse_args()

    if not args.input_file:
        args.input_file = r"E:\CAD数据集\项目\GFR_Dataset_Final\GFR_00971.step" 

    if not os.path.exists(args.input_file):
        print(f"错误：文件不存在 '{args.input_file}'")
        sys.exit(1)

    print(f"正在加载 STEP 文件: {args.input_file}...")
    try:
        compound = Compound.load_from_step(args.input_file)
    except Exception as e:
        print(f"错误：无法加载 STEP 文件 '{args.input_file}'。错误信息: {e}")
        sys.exit(1)

    compound = compound.scale_to_unit_box()

    shape_to_process = None
    solids = list(compound.solids())
    if solids:
        print(f"在文件中找到 {len(solids)} 个 Solid。将处理第一个。")
        shape_to_process = solids[0]
    else:
        shells = list(compound.shells())
        if shells:
            print(f"未找到 Solid，但在文件中找到 {len(shells)} 个 Shell。将处理第一个。")
            shape_to_process = shells[0]
        else:
            print("错误：在文件中既没有找到 Solid 也没有找到 Shell。无法进行可视化。")
            sys.exit(1)

    # --- 【核心修改】在这里指定你想要可视化的面的索引 ---
    # 例如: [149, 220, 236, 240]
    # 如果想可视化所有面，请将其设置为 None
    FACES_TO_HIGHLIGHT = [22]

    # 初始化查看器
    v = Viewer(backend="pyqt5")

    # 调用核心可视化函数，并传入要高亮的面索引列表
    visualize_control_points(v, shape_to_process, indices_to_show=FACES_TO_HIGHLIGHT)

    # 调整视角并显示窗口
    v.fit()
    v.show()