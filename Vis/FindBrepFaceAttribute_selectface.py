import sys
import argparse

# --- occwl 库用于加载、图构建和显示 ---
from occwl.compound import Compound
from occwl.viewer import Viewer
from occwl.graph import face_adjacency
from occwl.face import Face

# --- python-occ 库用于几何属性计算 ---
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopoDS import topods_Face, topods_Edge
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
                              GeomAbs_BSplineSurface)


import shutup
shutup.please()
# ===============================================================
#    函数定义部分 (结合了两个脚本的辅助函数)
# ===============================================================

def process_and_print_face_properties(face: Face, face_id: int):
    """提取单个面的几何属性并打印到控制台。"""
    topo_face = face.topods_shape()
    props = GProp_GProps()
    brepgprop.SurfaceProperties(topo_face, props)
    area = props.Mass()
    adaptor = BRepAdaptor_Surface(topo_face)
    face_type_enum = adaptor.GetType()
    type_name = {
        GeomAbs_Plane: "Plane", GeomAbs_Cylinder: "Cylinder", GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere", GeomAbs_Torus: "Torus", 
        GeomAbs_BezierSurface: "Bezier Surface", GeomAbs_BSplineSurface: "B-Spline Surface",
    }.get(face_type_enum, "Other")
    edge_explorer = TopExp_Explorer(topo_face, TopAbs_EDGE)
    num_edges = 0
    perimeter = 0.0
    while edge_explorer.More():
        edge = topods_Edge(edge_explorer.Current())
        edge_props = GProp_GProps()
        brepgprop.LinearProperties(edge, edge_props)
        perimeter += edge_props.Mass()
        num_edges += 1
        edge_explorer.Next()

    print("\n" + "=" * 35)
    print(f"▶️ 正在分析 面 (Face) 的索引: {face_id}")
    print("-" * 35)
    print(f"  - 面的类型 (Type)   : {type_name}")
    print(f"  - 面积 (Area)        : {area:.4f} mm^2")
    print(f"  - 边的数量 (Edges)   : {num_edges}")
    print(f"  - 周长 (Perimeter)   : {perimeter:.4f} mm")
    print("=" * 35)

def display_all_faces(viewer, face_obj_list, highlight_indices=None, highlight_color=(1, 0, 0), default_color=(0.8, 0.8, 0.8)):
    """将所有面绘制到viewer，高亮指定的面。"""
    highlight_indices = highlight_indices or []
    for idx, face in enumerate(face_obj_list):
        if idx in highlight_indices:
            viewer.display(face, color=highlight_color)
        else:
            # 让未选中的面半透明，以突出高亮面
            viewer.display(face, color=default_color, transparency=0.7)

def display_all_edges(viewer, edge_obj_list, default_color=(0.3, 0.3, 0.3)):
    """将所有边绘制到viewer，让模型轮廓更清晰。"""
    for edge in edge_obj_list:
        viewer.display(edge, color=default_color)

# ===============================================================
#    主程序 (__main__)
# ===============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="分析并可视化STEP文件中的指定面。"
    )
    parser.add_argument("--step_file", type=str,default = r"D:\CAD数据集\项目\GFR_Dataset_Final\GFR_02664.step", help="输入的STEP文件路径")
    parser.add_argument("--faces", nargs='+', type=int, help="要分析和高亮的面索引列表，例如: --faces 42 100")
    args = parser.parse_args()

    # --- 1. 定义要处理和高亮的面索引列表 ---
    if args.faces:
        face_indices_to_process = args.faces
        print(f"将处理通过命令行指定的面: {face_indices_to_process}")
    else:
        # ↓↓↓ 您可以在这里直接修改这个列表 ↓↓↓
        face_indices_to_process = [1, 10, 100]
        # ↑↑↑ 您可以在这里直接修改这个列表 ↑↑↑
        print(f"未从命令行指定面，将使用代码中定义的默认列表: {face_indices_to_process}")

    # --- 2. 加载模型并获取有序的面列表 ---
    try:
        print(f"\n正在加载 STEP 文件: {args.step_file}...")
        compound = Compound.load_from_step(args.step_file)
        
        # 使用 face_adjacency 来获取稳定有序的面列表，这比直接调用 .faces() 更健壮
        entity_to_process = next(compound.solids(), next(compound.shells(), None))
        if entity_to_process is None:
            raise RuntimeError("在文件中既没有找到 Solid 也没有找到 Shell。")
        
        print("加载成功！正在构建面邻接图以获取有序面列表...")
        graph = face_adjacency(entity_to_process)
        face_obj_list = [graph.nodes[i]["face"] for i in sorted(graph.nodes.keys())]
        edge_obj_list = [graph.edges[edge]["edge"] for edge in graph.edges]
        print(f"图构建完成，模型包含 {len(face_obj_list)} 个面。")

    except Exception as e:
        print(f"错误: 加载或处理STEP文件失败。错误信息: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. 处理指定的面并打印信息到控制台 ---
    print("\n====== 开始分析指定的面 (控制台输出) ======")
    for face_index in face_indices_to_process:
        if 0 <= face_index < len(face_obj_list):
            face_to_process = face_obj_list[face_index]
            process_and_print_face_properties(face_to_process, face_index)
        else:
            print(f"⚠️ 警告: 索引 {face_index} 超出范围 (模型总面数: {len(face_obj_list)})，已跳过。")
    print("====== 控制台输出处理完成 ======")

    # --- 4. 可视化模型并高亮指定的面 ---
    print("\n正在启动3D查看器...")
    v = Viewer(backend="pyqt5")
    
    # 绘制所有面，并高亮我们感兴趣的面
    display_all_faces(
        v, 
        face_obj_list, 
        highlight_indices=face_indices_to_process,
        highlight_color=(1, 0.2, 0.2),  # 高亮为醒目的红色
        default_color=(0.8, 0.8, 0.8)
    )

    # 绘制所有边，让模型轮廓更清晰
    display_all_edges(v, edge_obj_list)

    print("查看器已准备就绪。关闭窗口以结束程序。")
    v.fit()
    v.show()

    print("\n程序执行完毕。")