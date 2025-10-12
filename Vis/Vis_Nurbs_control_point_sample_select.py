# -*- coding: utf-8 -*-

"""
B-rep NURBS曲面控制点与均匀子采样点可视化脚本 - v5

功能:
    加载一个STEP文件，并对指定的面进行可视化。
    - 原始控制点显示为红色。
    - 原始控制多边形显示为黑色。
    - 经过均匀间隔选择出的原始控制点，将作为绿色大球叠加显示，
      用于直观地验证子采样算法的正确性。
"""

import os
import sys
import argparse
import numpy as np
import shutup

shutup.please()  # 静音一些不必要的警告信息

# MODIFIED: 不再需要 scipy
# try:
#     from scipy.ndimage import map_coordinates
# ...

# 核心 OCC 和 occwl 依赖
from occwl.viewer import Viewer
from occwl.compound import Compound
from occwl.edge import Edge
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

# 定义与数据处理脚本一致的采样目标尺寸
TARGET_CTRL_PNTS = 20


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
                             show_original_points=True,
                             show_hull=True,
                             show_subsampled_points=True, # MODIFIED: 重命名
                             original_point_color=(1, 0, 0),    # 红色: 原始控制点
                             subsampled_point_color=(0, 1, 0),  # 绿色: 被选中的点
                             point_scale=2,
                             subsampled_point_scale=4, # 让被选中的点更大更明显
                             hull_color=(0, 0, 0),             # 黑色
                             highlight_color=(0.1, 0.6, 1.0),
                             default_color=(0.8, 0.8, 0.8),
                             transparency=0.7):
    """
    在给定的viewer中绘制几何实体指定面的控制点、控制多边形以及均匀选择出的子采样点。
    """
    if not hasattr(shape, 'faces'):
        print("错误: 提供的对象没有 'faces' 属性。")
        return

    all_faces = list(shape.faces())
    
    faces_to_process = []
    if indices_to_show is None:
        print(f"开始处理全部 {len(all_faces)} 个面...")
        faces_to_process = list(enumerate(all_faces))
    else:
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

        viewer.display(face, color=highlight_color, transparency=transparency - 0.2)
        
        poles_occ = bspline_surface.Poles()
        poles_np = array_from_Array2OfPnt(poles_occ)
        
        # --- MODIFIED: 在此处执行与数据处理脚本相同的均匀子采样逻辑 ---
        if show_subsampled_points:
            print(f"  - 正在为面 {i} 计算均匀选择的控制点...")
            size_u, size_v = poles_np.shape[0], poles_np.shape[1]

            # 仅在原始点数大于目标点数时才进行子采样
            # U方向
            if size_u > TARGET_CTRL_PNTS:
                u_indices = np.linspace(0, size_u - 1, TARGET_CTRL_PNTS, dtype=int)
            else:
                u_indices = np.arange(size_u) # 否则保留所有
            # V方向
            if size_v > TARGET_CTRL_PNTS:
                v_indices = np.linspace(0, size_v - 1, TARGET_CTRL_PNTS, dtype=int)
            else:
                v_indices = np.arange(size_v) # 否则保留所有

            # 使用 np.ix_ 从原始网格中直接选择出对应的行和列
            subsampled_poles_np = poles_np[np.ix_(u_indices, v_indices)]
            
            # 显示被选中的点
            subsampled_flat = subsampled_poles_np.reshape(-1, 3)
            viewer.display_points(subsampled_flat, color=subsampled_point_color, scale=subsampled_point_scale)
            print(f"    ... {len(subsampled_flat)} 个被选择的点（绿色）已添加。")
        # --- 修改结束 ---

        if show_original_points:
            points_flat = poles_np.reshape(-1, 3)
            viewer.display_points(points_flat, color=original_point_color, scale=point_scale)
            print(f"  - {len(points_flat)} 个原始控制点（红色）已添加。")

        if show_hull:
            for row in poles_np:
                for j in range(len(row) - 1):
                    edge = create_edge_from_points(row[j], row[j+1])
                    if edge: viewer.display(edge, color=hull_color)
            
            for col in poles_np.T:
                for j in range(len(col) - 1):
                    edge = create_edge_from_points(col[j], col[j+1])
                    if edge: viewer.display(edge, color=hull_color)
    
    print("\n可视化处理完成。")
    print("图例: \n  - 红色小球: 原始控制点\n  - 绿色大球: 被均匀选择出的原始控制点\n  - 黑色线条: 原始控制网格")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("B-rep NURBS曲面控制点与均匀子采样点可视化脚本")
    parser.add_argument("--input_file", type=str, help="要可视化的STEP文件路径")
    args = parser.parse_args()

    if not args.input_file:
        # !!! 在这里修改你的默认STEP文件路径 !!!
        args.input_file = r"E:\CAD数据集\项目\GFR_Dataset_Final\GFR_00130.step" 

    if not os.path.exists(args.input_file):
        print(f"错误：文件不存在 '{args.input_file}'")
        sys.exit(1)

    print(f"正在加载 STEP 文件: {args.input_file}...")
    try:
        compound = Compound.load_from_step(args.input_file)
    except Exception as e:
        print(f"错误：无法加载 STEP 文件 '{args.gfr_path}'。错误信息: {e}")
        sys.exit(1)

    # 取消缩放，以观察原始坐标
    # compound = compound.scale_to_unit_box()

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

    # --- 【在这里指定你想要可视化的面的索引】---
    # 例如: [10, 15, 20]
    # 如果想可视化所有面，请将其设置为 None
    FACES_TO_HIGHLIGHT = [65]

    # 初始化查看器
    v = Viewer(backend="pyqt5")

    # 调用核心可视化函数
    visualize_control_points(v, shape_to_process, indices_to_show=FACES_TO_HIGHLIGHT)

    # 调整视角并显示窗口
    v.fit()
    v.show()