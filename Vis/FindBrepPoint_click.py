import sys
from collections import defaultdict
import argparse

# --- 使用底层 python-occ 库 ---
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX

# --- 仍然使用 occwl 进行加载和显示 ---
from occwl.compound import Compound
from occwl.viewer import Viewer
from occwl.face import Face

# --- 全局变量 ---
# 仍然需要这个列表来跟踪哪些面当前处于选中状态
selected_face_objects = [] 
# 新增：提前创建 Face 对象到其稳定索引的映射，以便在回调中随时使用
face_to_id_map = {}


def get_and_print_face_vertices(face: Face):
    """
    一个辅助函数，用于提取单个面的顶点坐标并将其打印到控制台。
    """
    face_id = face_to_id_map.get(face, -1) # -1 表示未找到，但理论上不会发生

    points = []
    topo_face = face.topods_shape() 

    exp = TopExp_Explorer(topo_face, TopAbs_VERTEX)
    while exp.More():
        topo_vertex = exp.Current()
        pt = BRep_Tool.Pnt(topo_vertex)
        xyz = (pt.X(), pt.Y(), pt.Z())
        points.append(xyz)
        exp.Next()
    
    unique_points = sorted(list(set(points)))

    # --- 格式化实时输出 ---
    print("\n" + "-" * 25)
    print(f"✅ 已选择 面 (Face) 的索引: {face_id}")
    if not unique_points:
        print("  - 该面没有独立的顶点。")
    for i, (x, y, z) in enumerate(unique_points):
        print(f"  顶点 {i+1}: (X={x:.4f}, Y={y:.4f}, Z={z:.4f})")
    print("-" * 25)


def on_select(shapes, x, y):
    """
    回调函数，现在负责实时处理选择、取消选择，并立即打印结果。
    """
    if not shapes:
        return
        
    shape = shapes[0]
    if isinstance(shape, Face):
        face = shape
        if face in selected_face_objects:
            # --- 这是取消选择的逻辑 ---
            selected_face_objects.remove(face)
            face_id = face_to_id_map.get(face, -1)
            print("\n" + "-" * 25)
            print(f"❌ 已取消选择 面 (Face) 的索引: {face_id}")
            print("-" * 25)
        else:
            # --- 这是选择的逻辑 ---
            selected_face_objects.append(face)
            # 立即调用函数来提取并打印这个刚被选中的面的坐标
            get_and_print_face_vertices(face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从一个STEP文件中交互式地选择面，并实时导出其顶点的坐标。"
    )
    parser.add_argument("--step_file", type=str,default = r"D:\CAD数据集\项目\GFR_Dataset_Final\GFR_02617.step", help="输入的STEP文件路径")
    args = parser.parse_args()

    try:
        print(f"正在加载 STEP 文件: {args.step_file}...")
        shape = Compound.load_from_step(args.step_file)
        all_faces = list(shape.faces())
        
        # --- 核心修改：在程序开始时就构建好 Face 到 ID 的映射 ---
        face_to_id_map = {face: i for i, face in enumerate(all_faces)}
        
        print(f"加载成功！模型包含 {len(all_faces)} 个面。")
    except Exception as e:
        print(f"错误: 加载STEP文件失败。请检查文件路径或文件是否损坏。错误信息: {e}")
        sys.exit(1)

    viewer = Viewer(backend="pyqt5")
    
    print("正在将面渲染到查看器中...")
    for face in all_faces:
        viewer.display(face, color=(0.7, 0.7, 0.8), transparency=0.5)
    print("渲染完成。")
    
    viewer.on_select(on_select)

    print("\n" + "="*60)
    print("操作指南:")
    print("1. 一个包含您模型的交互式窗口已经打开。")
    print("2. 使用鼠标左键点击任意一个面来选中它。")
    print("3. 【新】选中后，顶点的坐标会立即显示在下方的控制台中。")
    print("4. 再次点击已选中的面可以取消选择。")
    print("5. 完成所有操作后，请直接【关闭】查看器窗口以退出程序。")
    print("="*60 + "\n")
    
    viewer.fit()
    viewer.show()

    # --- 主程序在窗口关闭后不再需要进行数据处理 ---
    print("\n查看器已关闭。程序执行完毕。")
    if selected_face_objects:
        final_selected_ids = sorted([face_to_id_map.get(f) for f in selected_face_objects])
        print(f"最终处于选中状态的面索引为: {final_selected_ids}")