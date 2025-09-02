import sys
import argparse

# --- 使用底层 python-occ 库 ---
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.gp import gp_Pnt

# --- 仅使用 occwl 加载 STEP 文件 ---
from occwl.compound import Compound
from occwl.face import Face

# --- 全局变量 ---
# 用于存储全局顶点索引的映射字典
vertex_coord_str_to_global_id = {}

def _quantize_point(p: gp_Pnt) -> str:
    """
    将一个 gp_Pnt 对象的坐标量化成一个唯一的字符串键。
    """
    return f"{p.X():.3f}_{p.Y():.3f}_{p.Z():.3f}"

def process_and_print_face_info(face: Face, face_id: int):
    """
    一个辅助函数，用于提取单个面的顶点坐标及其自定义的全局索引并打印。
    """

    points_with_ids = []
    topo_face = face.topods_shape() 

    exp = TopExp_Explorer(topo_face, TopAbs_VERTEX)
    while exp.More():
        # ★★★ 核心修改：直接使用 exp.Current()，不再用 topods_Vertex() 包裹 ★★★
        topo_vertex = exp.Current()
        pt = BRep_Tool.Pnt(topo_vertex)
        
        # 使用我们自定义的、可靠的坐标量化方法来查找全局索引
        coord_key = _quantize_point(pt)
        global_id = vertex_coord_str_to_global_id.get(coord_key, -1)
        
        xyz = (pt.X(), pt.Y(), pt.Z())
        points_with_ids.append((global_id, xyz))
        
        exp.Next()
    
    # 去重并按全局ID排序
    unique_points = sorted(list(set(points_with_ids)), key=lambda item: item[0])

    # --- 格式化输出 ---
    print("\n" + "-" * 55)
    print(f"▶️ 正在处理 面 (Face) 的索引: {face_id}")
    if not unique_points:
        print("  - 该面没有独立的顶点。")
    for i, (global_id, (x, y, z)) in enumerate(unique_points):
        print(f"  顶点 {i+1} (全局索引: {global_id}): (X={x:.4f}, Y={y:.4f}, Z={z:.4f})")
    print("-" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从STEP文件中提取指定面的顶点的全局索引和坐标。"
    )
    # ★★★ 修改：让 step_file 参数变为可选，并设置默认值 ★★★
    parser.add_argument("step_file", type=str, nargs='?', default=r"D:\CAD数据集\项目\GFR_Dataset_Final\GFR_02664.step", help="输入的STEP文件路径")
    parser.add_argument("--faces", nargs='+', type=int, help="要处理的面索引列表，例如: --faces 2 5 222")
    args = parser.parse_args()

    if args.faces:
        face_indices_to_process = args.faces
        print(f"将处理通过命令行指定的面: {face_indices_to_process}")
    else:
        face_indices_to_process = [2, 5, 222, 223]
        print(f"未从命令行指定面，将使用代码中定义的默认列表: {face_indices_to_process}")

    try:
        print(f"\n正在加载 STEP 文件: {args.step_file}...")
        shape = Compound.load_from_step(args.step_file)
        all_faces = list(shape.faces())
        print(f"加载成功！模型包含 {len(all_faces)} 个面。")

        print("正在为整个模型构建全局顶点索引...")
        global_vertex_counter = 0
        exp = TopExp_Explorer(shape.topods_shape(), TopAbs_VERTEX)
        while exp.More():
            # ★★★ 核心修改：直接使用 exp.Current() ★★★
            vertex = exp.Current()
            point = BRep_Tool.Pnt(vertex)
            coord_key = _quantize_point(point)
            if coord_key not in vertex_coord_str_to_global_id:
                vertex_coord_str_to_global_id[coord_key] = global_vertex_counter
                global_vertex_counter += 1
            exp.Next()
        print(f"索引构建完成！模型中共有 {global_vertex_counter} 个唯一的顶点。")

    except Exception as e:
        print(f"错误: 加载或处理STEP文件失败。错误信息: {e}")
        sys.exit(1)

    print("\n====== 开始处理指定的面 ======")
    for face_index in face_indices_to_process:
        if 0 <= face_index < len(all_faces):
            face_to_process = all_faces[face_index]
            process_and_print_face_info(face_to_process, face_index)
        else:
            print("\n" + "-" * 55)
            print(f"⚠️ 警告: 索引 {face_index} 超出范围 (模型总面数: {len(all_faces)})，已跳过。")
            print("-" * 55)

    print("\n====== 处理完成 ======")