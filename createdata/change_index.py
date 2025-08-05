import os
from collections import defaultdict

from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE,TopAbs_VERTEX
from OCC.Core.TopoDS import topods_Face, topods_Edge, topods
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_LinearProperties, brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceType
import json

from tqdm import tqdm  # 正确导入 tqdm 类

from OCC.Core._TopAbs import TopAbs_SHAPE

def read_stp(file_path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status == IFSelect_RetDone:  # 检查文件是否成功读取
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape
    else:
        raise Exception("Failed to read STEP file")

def find_original_shape(face, shapes):
    # 确定该表面属于哪个原始模型
    for original_shape in shapes:
        # 遍历原始模型中的每个表面
        explorer = TopExp_Explorer(original_shape, TopAbs_FACE)
        while explorer.More():
            f = explorer.Current()
            if face.IsEqual(f):
                return original_shape
            explorer.Next()
    return None
def get_surface_index(shape, face):
    # 获取表面在形状中的索引
    index = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        f = explorer.Current()
        if face.IsEqual(f):
            return index
        index += 1
        explorer.Next()
    return None
def getNum(y,x):
    if(y==3):
        return x
    else:
        return y

def get_faces_info(shape):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces_info = []
    index = 0
    vertex_to_faces = defaultdict(set)
    while explorer.More():
        face = topods.Face(explorer.Current())

        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()  # 面积

        adaptor = BRepAdaptor_Surface(face)
        face_type = adaptor.GetType()

        type_name = {
            GeomAbs_Plane: "Plane",
            GeomAbs_Cylinder: "Cylinder",
            GeomAbs_Cone: "Cone",
            GeomAbs_Sphere: "Sphere",
            GeomAbs_Torus: "Torus",
            GeomAbs_BezierSurface: "BezierSurface",
            GeomAbs_BSplineSurface: "BSplineSurface",
            GeomAbs_SurfaceType: "Other"
        }.get(face_type, "Unknown")

        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        num_edges = 0
        perimeter = 0.0
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            edge_props = GProp_GProps()
            brepgprop.LinearProperties(edge, edge_props)
            perimeter += edge_props.Mass()  # 累加边缘长度得到周长
            num_edges += 1
            edge_explorer.Next()

        faces_info.append((index, type_name, area, num_edges, perimeter))
        index += 1
        vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
        vertices_info = []
        while vertex_explorer.More():
            vertex = topods.Vertex(vertex_explorer.Current())
            vertex_point = BRep_Tool.Pnt(vertex)  # 正确获取顶点的坐标
            vertex_coords = (vertex_point.X(), vertex_point.Y(), vertex_point.Z())
            vertices_info.append(vertex_coords)
            vertex_to_faces[vertex].add(index)  # 记录顶点所属的面
            vertex_explorer.Next()

        explorer.Next()

    return faces_info, vertex_to_faces


def nearly_equal(a, b, threshold=1e-3):
    """Check if two floating point numbers are close enough based on a relative threshold."""
    return abs(a - b) <= threshold * max(abs(a), abs(b))
def export_face_to_model_mapping(mapping, file_path):
    with open(file_path, 'w') as f:
        json.dump(mapping, f, indent=4)
def export_unmatched_faces(faces_info, file_path):
    """将未匹配的面信息导出到JSON文件"""
    with open(file_path, 'w') as f:
        json.dump(faces_info, f, indent=4)



def process_base_shape(base_file_path, part_files, output_dir):
    base_shape = read_stp(base_file_path)
    base_faces_info, vertex_to_faces = get_faces_info(base_shape)

    parts_shapes = [read_stp(part_file) for part_file in part_files]
    parts_faces_info = [get_faces_info(shape)[0] for shape in parts_shapes]

    # 用于记录匹配结果
    mapping = {}  # 每个基础模型面最多匹配一个部件模型面
    unmatched_base_faces = []  # 未匹配的基础模型面信息
    unmatched_part_faces = [[] for _ in parts_faces_info]  # 为每个部件模型存储未匹配的面信息

    MATCH_THRESHOLD = 0.01  # 匹配分数的阈值，用于面积和周长
    VERTEX_TOLERANCE = 1  # 顶点数差异容忍度

    def nearly_equal(a, b, threshold=MATCH_THRESHOLD):
        """Check if two floating point numbers are close enough based on an absolute threshold."""
        return abs(a - b) <= threshold

    def nearly_equal_loose(a, b, tolerance=1):
        """Check if two floating point numbers are close enough when rounded to integers."""
        return abs(int(round(a)) - int(round(b))) <= tolerance

    # 第一阶段：宽松匹配
    for i, base_face in enumerate(base_faces_info):
        matched = False
        for part_index, part_faces_info in enumerate(parts_faces_info):
            for part_face in part_faces_info:
                # 检查面积和周长是否在阈值范围内
                if (nearly_equal(base_face[2], part_face[2]) and
                    nearly_equal(base_face[4], part_face[4])):
                    # 检查面类型和顶点数
                    type_match = (base_face[1] == part_face[1] or part_face[1] == "Unknown")
                    vertex_match = abs(base_face[3] - part_face[3]) <= VERTEX_TOLERANCE

                    # 允许类型或顶点数中的一个不匹配
                    if type_match or vertex_match:
                        mapping[i] = part_index # 记录匹配的部件模型索引
                        matched = True
                        break  # 找到匹配后，停止匹配其他部件模型的面
            if matched:
                break

        # 如果没有找到匹配，记录为未匹配
        if not matched:
            unmatched_base_faces.append(base_face)

    # 第二阶段：严格匹配（针对未匹配的基础模型面）
    for base_face in unmatched_base_faces[:]:  # 遍历未匹配面的副本
        matched = False
        for part_index, part_faces_info in enumerate(parts_faces_info):
            for part_face in part_faces_info:
                # 检查面类型和边数量是否一致
                if (base_face[1] == part_face[1] and
                    base_face[3] == part_face[3]):
                    # 检查面积和周长是否在整数级别一致，允许一定的误差范围
                    if (nearly_equal_loose(base_face[2], part_face[2]) and
                        nearly_equal_loose(base_face[4], part_face[4])):
                        mapping[base_faces_info.index(base_face)] = part_index   # 记录匹配的部件模型索引
                        matched = True
                        unmatched_base_faces.remove(base_face)  # 从未匹配列表中移除
                        break  # 找到匹配后，停止匹配其他部件模型的面
            if matched:
                break

    # 在严格匹配之后，将所有未匹配的基础模型面的索引加入到mapping中，并设置label为0
    for i, base_face in enumerate(base_faces_info):
        if i not in mapping:
            mapping[i] = 0  # 将未匹配的基础模型面的label设置为0

    # 检查每个部件模型中未匹配的面
    for part_index, part_faces_info in enumerate(parts_faces_info):
        for part_face in part_faces_info:
            matched = False
            for base_face in base_faces_info:
                if (nearly_equal(base_face[2], part_face[2], MATCH_THRESHOLD) and
                    nearly_equal(base_face[4], part_face[4], MATCH_THRESHOLD)):
                    type_match = (base_face[1] == part_face[1] or part_face[1] == "Unknown")
                    vertex_match = abs(base_face[3] - part_face[3]) <= VERTEX_TOLERANCE
                    if type_match or vertex_match:
                        matched = True
                        break
            if not matched:
                unmatched_part_faces[part_index].append(part_face)

    # 比较未匹配的面数量
    unmatched_base_count = len(unmatched_base_faces)
    unmatched_part_counts = [len(part_faces) for part_faces in unmatched_part_faces]

    # 输出比较结果到控制台
    # print(f"基础模型未匹配的面数量: {unmatched_base_count}")
    for part_index, count in enumerate(unmatched_part_counts):
        pass
        # print(f"部件模型 {os.path.basename(part_files[part_index])} 未匹配的面数量: {count}")

    # 如果未匹配的面数量大于10，不存储未匹配信息，并删除原零件文件
    if unmatched_base_count > 10:
        os.remove(base_file_path)
        # print(f"基础模型未匹配的面数量超过10，不存储未匹配信息，并删除原基础模型文件: {base_file_path}")
    # else:
    #     # 导出未匹配的基础模型面信息
    #     unmatched_base_file = os.path.join(output_dir, os.path.splitext(os.path.basename(base_file_path))[0] + '_unmatched_base.json')
    #     export_unmatched_faces(unmatched_base_faces, unmatched_base_file)
    #
    # # 如果需要，可以将比较结果写入文件
    # comparison_result_file = os.path.join(output_dir, os.path.splitext(os.path.basename(base_file_path))[0] + '_comparison_result.txt')
    # with open(comparison_result_file, 'w') as f:
    #     f.write(f"基础模型未匹配的面数量: {unmatched_base_count}\n")
    #     for part_index, count in enumerate(unmatched_part_counts):
    #         f.write(f"部件模型 {os.path.basename(part_files[part_index])} 未匹配的面数量: {count}\n")

    # 导出匹配结果
    output_json = os.path.join(output_dir, os.path.splitext(os.path.basename(base_file_path))[0] + '.json')
    export_face_to_model_mapping(mapping, output_json)
    # print("ok")


if __name__ == "__main__":
    base_step_folder = r"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\data\step\step2"  # 包含基础模型的文件夹路径
    part_files = [
        r"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\原型\底座1.STEP",
        r"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\原型\螺钉1.STEP",
        r"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\原型\内六角1.STEP",
        r"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\原型\凸台1.STEP",
    ]
    output_directory = r"C:\Users\20268\Desktop\Project\ProcessData\Data\CBF\data\label"  # 输出JSON文件的目录

    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    all_files = os.listdir(base_step_folder)
    # 遍历STEP文件夹下的所有STEP文件

    # 使用 tqdm 包装现有的 for 循环
    for filename in tqdm(all_files, desc="Processing STEP Files", unit="file"):
        if filename.lower().endswith('.step'):
            base_file_path = os.path.join(base_step_folder, filename)

            # 构造对应的 .json 文件路径
            json_filename = os.path.splitext(filename)[0] + ".json"  # 将 .step 替换为 .json
            json_file_path = os.path.join(output_directory, json_filename)

            # 检查对应的 .json 文件是否已存在
            if os.path.exists(json_file_path):
                print(f"Skipping already processed file: {filename}")
                continue  # 跳过已存在的文件

            process_base_shape(base_file_path, part_files, output_directory)

    # for filename in os.listdir(base_step_folder):
    #     if filename.lower().endswith('.step'):
    #         base_file_path = os.path.join(base_step_folder, filename)
    #         process_base_shape(base_file_path, part_files, output_directory)



