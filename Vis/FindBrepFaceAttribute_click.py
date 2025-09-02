import sys
import argparse

# --- occwl 库用于加载和显示 ---
from occwl.compound import Compound
from occwl.viewer import Viewer
from occwl.face import Face

# --- python-occ 库用于几何属性计算 (核心逻辑来自您的参考代码) ---
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


# --- 全局变量 ---
# 用于跟踪哪些面当前处于选中状态
selected_face_objects = [] 
# 提前创建 Face 对象到其稳定索引的映射
face_to_id_map = {}

def get_and_print_face_properties(face: Face):
    """
    一个辅助函数，用于提取单个面的几何属性并将其打印到控制台。
    此函数的核心逻辑完全基于您提供的 GeometricPropertyAnalyzer._get_face_properties 方法。
    """
    face_id = face_to_id_map.get(face, -1)
    topo_face = face.topods_shape()

    # 1. 计算面积
    props = GProp_GProps()
    brepgprop.SurfaceProperties(topo_face, props)
    area = props.Mass()

    # 2. 获取面的类型
    adaptor = BRepAdaptor_Surface(topo_face)
    face_type_enum = adaptor.GetType()
    type_name = {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BezierSurface: "Bezier Surface",
        GeomAbs_BSplineSurface: "B-Spline Surface",
    }.get(face_type_enum, "Other")

    # 3. 计算边的数量和周长
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

    # --- 格式化实时输出 ---
    print("\n" + "=" * 35)
    print(f"✅ 已选择 面 (Face) 的索引: {face_id}")
    print("-" * 35)
    print(f"  - 面的类型 (Type)   : {type_name}")
    print(f"  - 面积 (Area)        : {area:.4f} mm^2")
    print(f"  - 边的数量 (Edges)   : {num_edges}")
    print(f"  - 周长 (Perimeter)   : {perimeter:.4f} mm")
    print("=" * 35)


def on_select(shapes, x, y):
    """
    回调函数，负责实时处理选择、取消选择，并立即调用属性分析函数。
    """
    if not shapes:
        return
        
    shape = shapes[0]
    if isinstance(shape, Face):
        face = shape
        if face in selected_face_objects:
            # 取消选择逻辑
            selected_face_objects.remove(face)
            face_id = face_to_id_map.get(face, -1)
            print("\n" + "-" * 25)
            print(f"❌ 已取消选择 面 (Face) 的索引: {face_id}")
            print("-" * 25)
        else:
            # 选择逻辑
            selected_face_objects.append(face)
            # 立即调用函数来提取并打印这个刚被选中的面的属性
            get_and_print_face_properties(face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从STEP文件中交互式地选择面，并实时分析其几何属性。"
    )
    parser.add_argument("--step_file", type=str,default = r"D:\CAD数据集\项目\GFR_Dataset_Final\GFR_02664.step", help="输入的STEP文件路径")
    args = parser.parse_args()

    try:
        print(f"正在加载 STEP 文件: {args.step_file}...")
        shape = Compound.load_from_step(args.step_file)
        all_faces = list(shape.faces())
        
        # 在程序开始时就构建好 Face 到 ID 的映射
        face_to_id_map = {face: i for i, face in enumerate(all_faces)}
        
        print(f"加载成功！模型包含 {len(all_faces)} 个面。")
    except Exception as e:
        print(f"错误: 加载STEP文件失败。请检查文件路径或文件是否损坏。错误信息: {e}", file=sys.stderr)
        sys.exit(1)

    viewer = Viewer(backend="pyqt5")
    
    print("正在将面渲染到查看器中...")
    for face in all_faces:
        viewer.display(face, color=(0.7, 0.7, 0.8), transparency=0.2)
    print("渲染完成。")
    
    viewer.on_select(on_select)

    print("\n" + "="*60)
    print("操作指南:")
    print("1. 一个包含您模型的交互式窗口已经打开。")
    print("2. 使用鼠标左键点击任意一个面来选中它。")
    print("3. 【新】选中后，该面的几何属性会立即显示在下方的控制台中。")
    print("4. 再次点击已选中的面可以取消选择。")
    print("5. 完成所有操作后，请直接【关闭】查看器窗口以退出程序。")
    print("="*60 + "\n")
    
    viewer.fit()
    viewer.show()

    print("\n查看器已关闭。程序执行完毕。")