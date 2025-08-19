import os
import json
import argparse
from pathlib import Path
import re
import logging  # 导入 logging 模块
from collections import defaultdict
import shutup
shutup.please()


# 设置日志记录
log_file = "process_log.txt"  # 日志文件
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(log_file, mode='w', encoding='utf-8')  # 输出到文件
    ]
)

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
    """用于分析B-Rep模型的几何属性，为每个面提取一个属性元组。"""
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
            # 这里抛出的异常将在调用处被捕获
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
    """根据两阶段几何属性匹配算法，查找两个模型中相同的面。"""
    base_faces = base_analyzer.faces_info
    feature_faces = feature_analyzer.faces_info
    
    MATCH_THRESHOLD = 0.01
    VERTEX_TOLERANCE = 1
    
    def nearly_equal(a, b, threshold=MATCH_THRESHOLD):
        return abs(a - b) <= threshold

    matched_base_face_ids = set()
    available_feature_faces = list(feature_faces)

    unmatched_base_faces_stage1 = []
    for base_face in base_faces:
        base_id, base_type, base_area, base_edges, base_perimeter = base_face
        
        found_match = False
        for feature_face in available_feature_faces:
            _, f_type, f_area, f_edges, f_perimeter = feature_face
            
            if nearly_equal(base_area, f_area) and nearly_equal(base_perimeter, f_perimeter):
                type_match = (base_type == f_type)
                edge_match = abs(base_edges - f_edges) <= VERTEX_TOLERANCE
                
                if type_match or edge_match:
                    matched_base_face_ids.add(base_id)
                    available_feature_faces.remove(feature_face)
                    found_match = True
                    break
        
        if not found_match:
            unmatched_base_faces_stage1.append(base_face)

    for base_face in unmatched_base_faces_stage1:
        base_id, base_type, base_area, base_edges, base_perimeter = base_face
        
        found_match = False
        for feature_face in available_feature_faces:
            _, f_type, f_area, f_edges, f_perimeter = feature_face
            
            if base_type == f_type and base_edges == f_edges:
                if nearly_equal(base_area, f_area) and nearly_equal(base_perimeter, f_perimeter):
                    matched_base_face_ids.add(base_id)
                    available_feature_faces.remove(feature_face)
                    found_match = True
                    break
    
    return list(matched_base_face_ids)

def scan_step_dir(step_dir: str):
    """扫描目录，返回基础文件和子特征文件的映射。"""
    logging.info(f"扫描目录: {step_dir}")
    bases = {}
    children = defaultdict(dict)

    BASE_RE = re.compile(r"^(GFR_\d{5})\.step$", re.IGNORECASE)
    CHILD_RE = re.compile(r"^(GFR_\d{5})_([A-Za-z0-9]+)\.step$", re.IGNORECASE)

    for f in Path(step_dir).iterdir():
        if not f.is_file() or f.suffix.lower() != ".step":
            continue
        stem = f.stem

        m_base = BASE_RE.match(f.name)
        if m_base:
            key = m_base.group(1)
            bases[key] = f.name
            continue

        m_child = CHILD_RE.match(f.name)
        if m_child:
            key = m_child.group(1)
            feat = m_child.group(2).lower()
            children[key][feat] = f.name

    logging.info(f"扫描到 {len(bases)} 个基础文件和 {len(children)} 个特征文件")
    return bases, children


# ---- 修改点 1: 在函数内部增加异常处理 ----
def generate_labels_for_base(base_filename: str, child_map_for_base: dict, step_dir: str, attribute_config: dict) -> dict:
    logging.info(f"正在处理基础文件: {base_filename}")
    
    base_path = Path(step_dir) / base_filename
    analyzer_base = GeometricPropertyAnalyzer()
    
    try:
        # 尝试加载基础模型
        analyzer_base.load_brep_from_step(str(base_path))
    except Exception as e:
        # 如果基础模型加载失败，记录错误并返回 None，表示处理失败
        logging.error(f"处理基础文件 {base_filename} 时发生严重错误，已跳过。错误信息: {e}")
        return None

    num_faces = analyzer_base.get_num_faces()
    labels = {str(i): 0 for i in range(num_faces)}
    logging.info(f"基础文件 {base_filename} 包含 {num_faces} 个面")

    for feat_name, child_filename in child_map_for_base.items():
        feat_key = feat_name.lower()
        if feat_key not in attribute_config['labels']:
            continue
        
        try:
            # 尝试加载和处理每个特征模型
            logging.info(f"处理特征: {feat_name}, 对应文件: {child_filename}")
            child_path = Path(step_dir) / child_filename
            analyzer_feature = GeometricPropertyAnalyzer()
            analyzer_feature.load_brep_from_step(str(child_path))
            
            matching_faces = find_matching_faces(analyzer_base, analyzer_feature)
            
            label_value = attribute_config['labels'][feat_key]
            logging.info(f"特征 '{feat_name}' 标签: {label_value}，匹配到 {len(matching_faces)} 个面")
            
            for face_id in matching_faces:
                labels[str(face_id)] = label_value
        except Exception as e:
            # 如果某个特征文件处理失败，只记录警告，然后继续处理下一个特征
            logging.warning(f"处理特征文件 {child_filename} 时发生错误，已跳过该特征。错误信息: {e}")
            continue
    
    return labels


def save_json_data(path_name: str, data: dict):
    logging.info(f"保存标签到文件: {path_name}")
    with open(path_name, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)
    logging.info(f"标签已成功保存到: {path_name}")

if __name__ == '__main__':
    logging.info("脚本开始执行")
    
    parser = argparse.ArgumentParser(description="为STEP文件生成特征标签。")
    parser.add_argument("--input_dir", type=str, default = "/data_hdd/dataset/GFR_Dataset_Final", help="包含STEP数据集的输入目录。")
    parser.add_argument("--output_dir", type=str, default = "/data_hdd/dev01/dyk/dyk_data/GFR_dataset_label_my", help="用于保存输出标签JSON文件的目录。")
    parser.add_argument("--config_file", type=str, default = "/data_hdd/dev01/dyk/utils/attribute_config.json", help="attribute_config.json 文件的路径。")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"输入目录: {args.input_dir}")
    logging.info(f"输出目录: {args.output_dir}")
    logging.info(f"配置文件: {args.config_file}")
    
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        if 'labels' not in config:
            raise KeyError("'labels' 键在配置文件中未找到。")
    except Exception as e:
        logging.error(f"错误: 无法加载或解析配置文件 '{args.config_file}'. 错误信息: {e}")
        exit(1)

    logging.info("扫描STEP文件夹")
    bases, children = scan_step_dir(args.input_dir)
    
    for base_key, base_file in bases.items():
        output_json = output_path / f"{base_key}.json"
        if output_json.exists():
            logging.info(f"文件 {output_json} 已存在，跳过处理。")
            continue
        
        child_map = children.get(base_key, {})
        filtered_child_map = {
            feat: fname
            for feat, fname in child_map.items()
            if feat.lower() in config['labels']
        }
        
        logging.info(f"生成标签: {base_key}")
        label_dict = generate_labels_for_base(base_file, filtered_child_map, args.input_dir, config)
        
        # ---- 修改点 2: 检查 generate_labels_for_base 的返回值 ----
        # 如果返回的是 None，说明处理失败，直接跳到下一个循环
        if label_dict is None:
            continue
        
        save_json_data(output_json, label_dict)
        logging.info(f"{base_key} 标签已生成并保存至 {output_json}")
    
    logging.info("脚本执行完成")