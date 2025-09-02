import sys
import argparse

# --- OCC/OCCWL 核心库 ---
from occwl.compound import Compound
from occwl.viewer import Viewer
from occwl.graph import face_adjacency
from occwl.face import Face
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.gp import gp_Pnt

# --- 全局变量 ---
# 用于存储全局顶点索引的映射字典
vertex_coord_str_to_global_id = {}

# ===============================================================
#    函数定义部分 (结合了两个脚本的辅助函数)
# ===============================================================

def _quantize_point(p: gp_Pnt) -> str:
    """将一个 gp_Pnt 对象的坐标量化成一个唯一的字符串键。"""
    return f"{p.X():.3f}_{p.Y():.3f}_{p.Z():.3f}"

def process_and_print_face_info(face: Face, face_id: int):
    """提取单个面的顶点信息并打印到控制台。"""
    points_with_ids = []
    topo_face = face.topods_shape() 
    exp = TopExp_Explorer(topo_face, TopAbs_VERTEX)
    while exp.More():
        topo_vertex = exp.Current()
        pt = BRep_Tool.Pnt(topo_vertex)
        coord_key = _quantize_point(pt)
        global_id = vertex_coord_str_to_global_id.get(coord_key, -1)
        xyz = (pt.X(), pt.Y(), pt.Z())
        points_with_ids.append((global_id, xyz))
        exp.Next()
    
    unique_points = sorted(list(set(points_with_ids)), key=lambda item: item[0])

    print("\n" + "-" * 55)
    print(f"▶️ 正在处理 面 (Face) 的索引: {face_id}")
    if not unique_points:
        print("  - 该面没有独立的顶点。")
    for i, (global_id, (x, y, z)) in enumerate(unique_points):
        print(f"  顶点 {i+1} (全局索引: {global_id}): (X={x:.4f}, Y={y:.4f}, Z={z:.4f})")
    print("-" * 55)

def display_all_faces(viewer, face_obj_list, highlight_indices=None, highlight_color=(1,0,0), default_color=(0.8,0.8,0.8)):
    """将所有面绘制到viewer，高亮highlight_indices对应的面。"""
    highlight_indices = highlight_indices or []
    for idx, face in enumerate(face_obj_list):
        if idx in highlight_indices:
            viewer.display(face, color=highlight_color)
        else:
            viewer.display(face, color=default_color, transparency=0.5)

def display_all_edges(viewer, edge_obj_list, default_color=(0.3,0.3,0.3)):
    """将所有边绘制到viewer。"""
    for edge in edge_obj_list:
        viewer.display(edge, color=default_color)

# ===============================================================
#    主程序 (__main__)
# ===============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="处理并可视化STEP文件中的指定面。"
    )
    parser.add_argument("step_file", type=str, nargs='?', default=r"D:\CAD数据集\项目\GFR_Dataset_Final\GFR_02664.step", help="输入的STEP文件路径")
    parser.add_argument("--faces", nargs='+', type=int, help="要处理和高亮的面索引列表，例如: --faces 2 5 222")
    args = parser.parse_args()

    # --- 1. 定义要处理和高亮的面索引列表 ---
    if args.faces:
        face_indices_to_process = args.faces
        print(f"将处理通过命令行指定的面: {face_indices_to_process}")
    else:
        # ↓↓↓ 您可以在这里直接修改这个列表 ↓↓↓
        face_indices_to_process = [2, 5, 222, 223]
        # ↑↑↑ 您可以在这里直接修改这个列表 ↑↑↑
        print(f"未从命令行指定面，将使用代码中定义的默认列表: {face_indices_to_process}")

    # --- 2. 加载模型并进行预处理 ---
    try:
        print(f"\n正在加载 STEP 文件: {args.step_file}...")
        compound = Compound.load_from_step(args.step_file)
        
        # 寻找合适的处理实体 (Solid或Shell)，这是构建图所必需的
        entity_to_process = None
        solids = list(compound.solids())
        if solids:
            entity_to_process = solids[0]
        else:
            shells = list(compound.shells())
            if shells:
                entity_to_process = shells[0]
        
        if entity_to_process is None:
            raise RuntimeError("在文件中既没有找到 Solid 也没有找到 Shell。")
        
        print("加载成功！正在构建面邻接图...")
        graph = face_adjacency(entity_to_process)
        # 从图中获取稳定有序的面和边列表
        face_obj_list = [graph.nodes[i]["face"] for i in sorted(graph.nodes.keys())]
        edge_obj_list = [graph.edges[edge]["edge"] for edge in graph.edges]
        print(f"图构建完成，模型包含 {len(face_obj_list)} 个面。")

        print("正在为整个模型构建全局顶点索引...")
        global_vertex_counter = 0
        exp = TopExp_Explorer(compound.topods_shape(), TopAbs_VERTEX)
        while exp.More():
            vertex = exp.Current()
            point = BRep_Tool.Pnt(vertex)
            coord_key = _quantize_point(point)
            if coord_key not in vertex_coord_str_to_global_id:
                vertex_coord_str_to_global_id[coord_key] = global_vertex_counter
                global_vertex_counter += 1
            exp.Next()
        print(f"索引构建完成！模型中共有 {global_vertex_counter} 个唯一的顶点。")

    except Exception as e:
        print(f"错误: 加载或处理STEP文件失败。错误信息: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. 处理指定的面并打印信息 ---
    print("\n====== 开始处理指定的面 (控制台输出) ======")
    for face_index in face_indices_to_process:
        if 0 <= face_index < len(face_obj_list):
            face_to_process = face_obj_list[face_index]
            process_and_print_face_info(face_to_process, face_index)
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
        highlight_color=(1, 0, 0),  # 高亮为红色
        default_color=(0.8, 0.8, 0.8)
    )

    # 绘制所有边，让模型轮廓更清晰
    display_all_edges(v, edge_obj_list)

    print("查看器已准备就绪。关闭窗口以结束程序。")
    v.fit()
    v.show()

    print("\n程序执行完毕。")