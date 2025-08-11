# -*- coding: utf-8 -*-
"""
label_generator.py

一个独立的脚本，用于为B-Rep（STEP）模型自动生成面部语义标签。

工作原理:
该脚本通过比较一个基础模型和一个或多个“特征”模型来工作。它通过分析构成每个面的
顶点集（“顶点指纹”）来识别两个模型中相同的面。如果基础模型中的一个面与
“rib”（筋）特征模型中的某个面匹配，那么该面就会被标记为“rib”。

输入:
1.  一个基础STEP文件 (例如: 'GFR_00001.step')。
2.  一个包含特征STEP文件的文件夹。特征文件应遵循命名约定：
    <基础文件名>_<特征名>.step (例如: 'GFR_00001_rib.step', 'GFR_00001_hole.step')。
3.  一个JSON配置文件，定义了特征名到整数标签的映射。

输出:
一个JSON文件，包含一个字典，键是基础模型中面的ID，值是对应的整数标签。

命令行用法示例:
python label_generator.py \
    --base_file /path/to/dataset/GFR_00001.step \
    --feature_dir /path/to/dataset/ \
    --config /path/to/attribute_config.json \
    --output /path/to/output/GFR_00001_labels.json
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

# 导入必要的OCC库
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_FACE
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape, topods_Face
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt


class BrepFeaturePointAnalyzer:
    """
    分析B-Rep模型的特征点（顶点），为每个面创建“顶点指纹”。
    这个类是自动打标签的核心引擎。
    """
    def __init__(self, tolerance_digits=1):
        """
        初始化分析器。
        
        参数:
            tolerance_digits (int): 点坐标量化时保留的小数位数。
                                   这是匹配顶点的关键容差。
        """
        self.point_id_to_coord = {}  # 点ID -> 坐标(gp_Pnt) 的映射
        self.point_id_to_faces = defaultdict(set)  # 点ID -> 拥有该点的面ID集合 的映射
        self.face_id_to_points = defaultdict(set)  # 面ID -> 该面包含的点ID集合（即“顶点指纹”）
        self.face_id_to_actual_face = {}  # 面ID -> 实际TopoDS_Face对象 的映射
        self.next_face_id = 0
        self.quantize_format = f"{{:.{tolerance_digits}f}}_{{:.{tolerance_digits}f}}_{{:.{tolerance_digits}f}}"

    def _quantize_point(self, p: gp_Pnt) -> str:
        """
        将点坐标根据容差量化，并生成唯一的字符串ID。
        这是为了处理浮点数精度问题，确保位置相近的点有相同的ID。
        """
        return self.quantize_format.format(p.X(), p.Y(), p.Z())

    def extract_face_vertices(self, face: TopoDS_Face) -> list:
        """从单个面提取所有顶点对象。"""
        vertices = []
        explorer = TopExp_Explorer(face, TopAbs_VERTEX)
        while explorer.More():
            vertex = explorer.Current()
            pnt = BRep_Tool.Pnt(vertex)
            vertices.append(pnt)
            explorer.Next()
        return vertices

    def load_brep_from_step(self, file_path: str):
        """
        从STEP文件加载B-Rep模型并分析所有面的特征点。
        """
        reader = STEPControl_Reader()
        status = reader.ReadFile(file_path)
        if status != IFSelect_RetDone:
            raise Exception(f"错误: 无法读取STEP文件 {file_path}")
        reader.TransferRoots()
        shape = reader.Shape()

        # 遍历所有面
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods_Face(explorer.Current())
            face_id = self.next_face_id
            self.next_face_id += 1
            self.face_id_to_actual_face[face_id] = face
            
            # 提取该面的所有顶点
            vertices = self.extract_face_vertices(face)
            for pnt in vertices:
                pid = self._quantize_point(pnt)
                
                # 记录点与面的双向关系
                self.point_id_to_faces[pid].add(face_id)
                self.face_id_to_points[face_id].add(pid)
                
                # 如果是新点，记录其坐标
                if pid not in self.point_id_to_coord:
                    self.point_id_to_coord[pid] = pnt
                    
            explorer.Next()
            
    def get_num_faces(self):
        return self.next_face_id

def find_matching_faces(analyzer1: BrepFeaturePointAnalyzer, 
                        analyzer2: BrepFeaturePointAnalyzer) -> list:
    """
    通过比较“顶点指纹”来查找两个B-Rep模型中相同的面。
    
    参数:
        analyzer1: 第一个（通常是基础）模型的分析器。
        analyzer2: 第二个（通常是特征）模型的分析器。
        
    返回:
        一个列表，包含在`analyzer1`中找到的匹配面的ID。
    """
    matching_face_ids_in_analyzer1 = []
    
    # 步骤1: 快速找出两个模型共有的特征点ID
    shared_point_ids = set(analyzer1.point_id_to_coord.keys()) & set(analyzer2.point_id_to_coord.keys())
    
    # 步骤2: 基于共享点，建立候选匹配对
    candidate_pairs = set()
    for pid in shared_point_ids:
        faces1 = analyzer1.point_id_to_faces.get(pid, set())
        faces2 = analyzer2.point_id_to_faces.get(pid, set())
        for f1 in faces1:
            for f2 in faces2:
                candidate_pairs.add((f1, f2))
    
    # 步骤3: 验证候选配对。只有当两个面的“顶点指纹”完全相同时，才认为是匹配的。
    processed_faces1 = set()
    for f1_id, f2_id in candidate_pairs:
        if f1_id in processed_faces1:
            continue
        
        points1 = analyzer1.face_id_to_points.get(f1_id, set())
        points2 = analyzer2.face_id_to_points.get(f2_id, set())
        
        if points1 and points1 == points2: # 确保指纹不为空且完全相同
            matching_face_ids_in_analyzer1.append(f1_id)
            processed_faces1.add(f1_id)

    print(f"    - 基础模型面数: {analyzer1.get_num_faces()}, 特征模型面数: {analyzer2.get_num_faces()}")
    print(f"    - 找到 {len(shared_point_ids)} 个共享顶点, {len(candidate_pairs)} 对候选面。")
    print(f"    - 最终确认 {len(matching_face_ids_in_analyzer1)} 个匹配的面。")
    return matching_face_ids_in_analyzer1

def generate_labels_for_file(base_file_path: str, feature_dir: str, attribute_config: dict) -> dict:
    """
    为单个基础STEP文件生成所有面的标签。
    
    参数:
        base_file_path: 基础STEP文件的完整路径。
        feature_dir: 存放特征STEP文件的目录。
        attribute_config: 从JSON加载的属性配置字典。
        
    返回:
        一个字典 {face_id: label}。
    """
    print(f"[*] 正在处理基础模型: {Path(base_file_path).name}")
    analyzer_base = BrepFeaturePointAnalyzer()
    analyzer_base.load_brep_from_step(base_file_path)
    
    # 初始化所有面的标签为0 (背景)
    num_faces = analyzer_base.get_num_faces()
    labels = {str(i): 0 for i in range(num_faces)}
    
    base_name_stem = Path(base_file_path).stem.lower()
    
    # 遍历特征目录，寻找相关的特征文件
    for filename in os.listdir(feature_dir):
        if not filename.lower().endswith(('.step', '.stp')):
            continue
        
        feature_path = Path(filename)
        if feature_path.stem.lower().startswith(base_name_stem + '_'):
            
            # 从文件名解析特征类型
            feature_key = feature_path.stem.lower().replace(base_name_stem + '_', '')
            
            if feature_key in attribute_config['labels']:
                label_value = attribute_config['labels'][feature_key]
                print(f"  [+] 发现并处理特征 '{feature_key}' (标签: {label_value})")
                
                # 为特征文件创建分析器
                analyzer_feature = BrepFeaturePointAnalyzer()
                analyzer_feature.load_brep_from_step(os.path.join(feature_dir, filename))
                
                # 查找匹配的面
                matching_faces = find_matching_faces(analyzer_base, analyzer_feature)
                
                # 更新标签
                for face_id in matching_faces:
                    labels[str(face_id)] = label_value
            else:
                print(f"  [!] 警告: 在配置文件中未找到特征 '{feature_key}' 的标签。")

    return labels


def save_json_data(path_name: str, data: dict):
    """将字典数据保存到JSON文件。"""
    with open(path_name, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="为B-Rep STEP模型自动生成面部语义标签。")
    parser.add_argument('--base_file', type=str, required=True, help='基础STEP文件的路径。')
    parser.add_argument('--feature_dir', type=str, required=True, help='包含特征STEP文件的目录。')
    parser.add_argument('--config', type=str, default="attribute_config.json",required=True, help='包含标签映射的JSON配置文件路径。')
    parser.add_argument('--output', type=str, required=True, help='输出标签JSON文件的路径。')
    
    args = parser.parse_args()
    
    
    # 加载配置文件
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        if 'labels' not in config:
            raise KeyError("'labels' 键在配置文件中未找到。")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"错误: 无法加载或解析配置文件 '{args.config}'. 错误信息: {e}")
        exit(1)
        
    # 生成标签
    try:
        final_labels = generate_labels_for_file(args.base_file, args.feature_dir, config)
    except Exception as e:
        print(f"在处理过程中发生严重错误: {e}")
        exit(1)

    # 保存结果
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_data(args.output, final_labels)
    
    print(f"\n[完成] 标签已成功生成并保存到: {args.output}")