# -*- coding: utf-8 -*-
"""
label_generator_refactored.py

一个重构后的脚本，用于为B-Rep模型自动生成面部标签。

核心逻辑:
该脚本结合了两种方法的优点：
1.  **框架**: 使用第一个脚本清晰的、基于分析器类 (Analyzer) 和配置文件驱动的框架。
2.  **匹配算法**: 使用第二个脚本的核心思想，即基于面的几何属性（面积、周长、类型、边数）进行两阶段匹配。

工作流程:
1.  为基础模型和每个特征模型创建一个`GeometricPropertyAnalyzer`实例，提取所有面的几何属性。
2.  调用一个统一的`find_matching_faces`函数，该函数内部实现了“宽松匹配”和“严格匹配”的两阶段逻辑。
3.  根据匹配结果，从配置文件中查找正确的标签并赋值。
4.  最终输出一个包含 {face_id: label} 映射的JSON文件。

命令行用法示例:
python label_generator_refactored.py \
    --base_file /path/to/dataset/A.step \
    --feature_dir /path/to/dataset/ \
    --config /path/to/attribute_config.json \
    --output /path/to/output/A_labels.json
"""

import os
import json
import argparse
from pathlib import Path

# 导入必要的OCC库
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopoDS import topods_Face, topods_Edge
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
                              GeomAbs_BSplineSurface, GeomAbs_SurfaceType)

class GeometricPropertyAnalyzer:
    """
    分析B-Rep模型的几何属性，为每个面提取一个属性元组。
    这个类取代了原脚本中零散的get_faces_info函数。
    """
    def __init__(self):
        self.faces_info = []  # 存储每个面的属性元组
        self.face_id_counter = 0

    def _get_face_properties(self, face: topods_Face):
        """为单个面提取几何属性。"""
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()

        adaptor = BRepAdaptor_Surface(face)
        face_type_enum = adaptor.GetType()
        type_name = {
            GeomAbs_Plane: "Plane",
            GeomAbs_Cylinder: "Cylinder",
            GeomAbs_Cone: "Cone",
            GeomAbs_Sphere: "Sphere",
            GeomAbs_Torus: "Torus",
            GeomAbs_BezierSurface: "BezierSurface",
            GeomAbs_BSplineSurface: "BSplineSurface",
        }.get(face_type_enum, "Other")

        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        num_edges = 0
        perimeter = 0.0
        while edge_explorer.More():
            edge = topods_Edge(edge_explorer.Current())
            edge_props = GProp_GProps()
            brepgprop.LinearProperties(edge, edge_props)
            perimeter += edge_props.Mass()
            num_edges += 1
            edge_explorer.Next()
        
        return (self.face_id_counter, type_name, area, num_edges, perimeter)

    def load_brep_from_step(self, file_path: str):
        """从STEP文件加载模型并提取所有面的几何属性。"""
        reader = STEPControl_Reader()
        status = reader.ReadFile(file_path)
        if status != IFSelect_RetDone:
            raise Exception(f"错误: 无法读取STEP文件 {file_path}")
        reader.TransferRoots()
        shape = reader.Shape()

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods_Face(explorer.Current())
            properties = self._get_face_properties(face)
            self.faces_info.append(properties)
            self.face_id_counter += 1
            explorer.Next()
            
    def get_num_faces(self):
        return len(self.faces_info)

def find_matching_faces(base_analyzer: GeometricPropertyAnalyzer, 
                        feature_analyzer: GeometricPropertyAnalyzer) -> list:
    """
    使用两阶段几何属性匹配算法，查找两个模型中相同的面。
    
    参数:
        base_analyzer: 基础模型的分析器。
        feature_analyzer: 特征模型的分析器。
        
    返回:
        一个列表，包含在`base_analyzer`中找到的匹配面的ID。
    """
    base_faces = base_analyzer.faces_info
    feature_faces = feature_analyzer.faces_info
    
    # 定义匹配参数
    MATCH_THRESHOLD = 0.01
    VERTEX_TOLERANCE = 1
    
    def nearly_equal(a, b, threshold=MATCH_THRESHOLD):
        return abs(a - b) <= threshold

    # 初始化
    matched_base_face_ids = set()
    available_feature_faces = list(feature_faces)

    # --- 阶段一：宽松匹配 ---
    unmatched_base_faces_stage1 = []
    for base_face in base_faces:
        base_id, base_type, base_area, base_edges, base_perimeter = base_face
        
        found_match = False
        for feature_face in available_feature_faces:
            _, f_type, f_area, f_edges, f_perimeter = feature_face
            
            # 检查面积和周长是否在阈值范围内
            if nearly_equal(base_area, f_area) and nearly_equal(base_perimeter, f_perimeter):
                # 检查面类型和边数
                type_match = (base_type == f_type)
                edge_match = abs(base_edges - f_edges) <= VERTEX_TOLERANCE
                
                # 只要类型或边数有一个匹配即可
                if type_match or edge_match:
                    matched_base_face_ids.add(base_id)
                    available_feature_faces.remove(feature_face) # 从可用列表中移除，避免重复匹配
                    found_match = True
                    break
        
        if not found_match:
            unmatched_base_faces_stage1.append(base_face)

    # --- 阶段二：严格匹配 ---
    # 仅针对在第一阶段未匹配上的基础面进行
    for base_face in unmatched_base_faces_stage1:
        base_id, base_type, base_area, base_edges, base_perimeter = base_face
        
        found_match = False
        for feature_face in available_feature_faces:
            _, f_type, f_area, f_edges, f_perimeter = feature_face
            
            # 严格要求类型和边数完全匹配
            if base_type == f_type and base_edges == f_edges:
                 # 面积和周长的要求可以保持不变，或根据需要调整
                if nearly_equal(base_area, f_area) and nearly_equal(base_perimeter, f_perimeter):
                    matched_base_face_ids.add(base_id)
                    available_feature_faces.remove(feature_face)
                    found_match = True
                    break
    
    print(f"    - 匹配完成。共找到 {len(matched_base_face_ids)} 个匹配的面。")
    return list(matched_base_face_ids)


def generate_labels_for_file(base_file_path: str, feature_dir: str, attribute_config: dict) -> dict:
    """为单个基础STEP文件生成所有面的标签。"""
    print(f"[*] 正在处理基础模型: {Path(base_file_path).name}")
    analyzer_base = GeometricPropertyAnalyzer()
    analyzer_base.load_brep_from_step(base_file_path)
    
    # 初始化所有面的标签为0 (背景)
    num_faces = analyzer_base.get_num_faces()
    labels = {str(i): 0 for i in range(num_faces)}
    
    base_name_stem = Path(base_file_path).stem.lower()
    
    # 对文件进行排序，确保处理顺序是确定的
    feature_files = sorted(os.listdir(feature_dir))
    
    for filename in feature_files:
        if not filename.lower().endswith(('.step', '.stp')):
            continue
        
        feature_path = Path(filename)
        if feature_path.stem.lower().startswith(base_name_stem + '_'):
            feature_key = feature_path.stem.lower().replace(base_name_stem + '_', '')
            
            if feature_key in attribute_config['labels']:
                label_value = attribute_config['labels'][feature_key]
                print(f"  [+] 发现并处理特征 '{feature_key}' (标签: {label_value})")
                
                analyzer_feature = GeometricPropertyAnalyzer()
                analyzer_feature.load_brep_from_step(os.path.join(feature_dir, filename))
                
                matching_faces = find_matching_faces(analyzer_base, analyzer_feature)
                
                for face_id in matching_faces:
                    face_id_str = str(face_id)
                    if labels[face_id_str] != 0 and labels[face_id_str] != label_value:
                        print(f"  [!!!] 警告: 面 {face_id_str} 被重复打标！")
                        print(f"      - 当前标签: {labels[face_id_str]}, 新标签: {label_value} (来自 {filename})")
                    labels[face_id_str] = label_value
            else:
                print(f"  [!] 警告: 在配置文件中未找到特征 '{feature_key}' 的标签。")

    return labels


def save_json_data(path_name: str, data: dict):
    """将字典数据保存到JSON文件。"""
    with open(path_name, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用几何属性匹配为B-Rep模型自动生成面部标签。")
    parser.add_argument('--base_file', type=str, required=True, help='基础STEP文件的路径。')
    parser.add_argument('--feature_dir', type=str, required=True, help='包含特征STEP文件的目录。')
    parser.add_argument('--config', type=str, required=True, help='包含标签映射的JSON配置文件路径。')
    parser.add_argument('--output', type=str, required=True, help='输出标签JSON文件的路径。')
    
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        if 'labels' not in config:
            raise KeyError("'labels' 键在配置文件中未找到。")
    except Exception as e:
        print(f"错误: 无法加载或解析配置文件 '{args.config}'. 错误信息: {e}")
        exit(1)
        
    try:
        final_labels = generate_labels_for_file(args.base_file, args.feature_dir, config)
    except Exception as e:
        print(f"在处理过程中发生严重错误: {e}")
        exit(1)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_data(args.output, final_labels)
    
    print(f"\n[完成] 标签已成功生成并保存到: {args.output}")