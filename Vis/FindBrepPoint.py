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
selected_faces_with_points = defaultdict(list)
selected_face_objects = []


def on_select(shapes, x, y):
    """回调函数，用于处理面选择事件。"""
    if not shapes:
        return
    shape = shapes[0]
    if isinstance(shape, Face):
        if shape in selected_face_objects:
            print(f"信息: 取消选择面。")
            selected_face_objects.remove(shape)
        else:
            print(f"信息: 已选择一个面！")
            selected_face_objects.append(shape)

def extract_vertex_coordinates_from_selected_faces(all_faces_in_shape):
    """
    遍历被选中的面，并使用 BRep_Tool.Pnt() 方式提取顶点坐标。
    """
    face_to_id_map = {face: i for i, face in enumerate(all_faces_in_shape)}

    for face in selected_face_objects:
        face_id = face_to_id_map.get(face)
        if face_id is None:
            continue

        points = []
        
        # --- 【核心修改】 ---
        # 从 occwl.Face 对象获取底层 TopoDS_Face 的正确方法是调用 .topods_shape()
        topo_face = face.topods_shape() 

        # 使用 TopExp_Explorer 遍历面上的所有顶点
        exp = TopExp_Explorer(topo_face, TopAbs_VERTEX)
        while exp.More():
            topo_vertex = exp.Current()
            pt = BRep_Tool.Pnt(topo_vertex)
            xyz = (pt.X(), pt.Y(), pt.Z())
            points.append(xyz)
            exp.Next()
        
        # 去重并排序
        unique_points = sorted(list(set(points)))
        selected_faces_with_points[face_id] = unique_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从一个STEP文件中交互式地选择面，并导出其顶点的坐标。"
    )
    parser.add_argument("--step_file", type=str,default = r"D:\CAD数据集\项目\GFR_Dataset_Final\GFR_02617_contact.step", help="输入的STEP文件路径")
    args = parser.parse_args()

    try:
        print(f"正在加载 STEP 文件: {args.step_file}...")
        shape = Compound.load_from_step(args.step_file)
        all_faces = list(shape.faces())
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
    print("3. 再次点击已选中的面可以取消选择。")
    print("4. 完成所有选择后，请直接【关闭】查看器窗口以继续执行程序。")
    print("="*60 + "\n")
    
    viewer.fit()
    viewer.show()

    print("\n查看器已关闭。正在处理您选择的面...")

    if not selected_face_objects:
        print("您没有选择任何面。程序退出。")
    else:
        extract_vertex_coordinates_from_selected_faces(all_faces)

        print(f"\n已成功提取 {len(selected_faces_with_points)} 个选中面的顶点坐标:")
        print("-" * 40)
        for face_id, points in sorted(selected_faces_with_points.items()):
            print(f"面 (Face) 的索引: {face_id}")
            if not points:
                print("  - 该面没有独立的顶点。")
            for i, (x, y, z) in enumerate(points):
                print(f"  顶点 {i+1}: (X={x:.4f}, Y={y:.4f}, Z={z:.4f})")
            print("-" * 40)