# -*- coding: utf-8 -*-
"""
生成说明：
- 输入：一个目录，包含 base STEP 文件（GFR_xxxxx.step）及其若干子特征 STEP（GFR_xxxxx_<feat>.step）
- 自动识别每个 base 拥有的子特征文件（仅限 attribute_config['lables'] 中存在的特征名）
- 通过公共顶点集合匹配面，并为匹配到的面写入对应 label（未匹配为 0）
- 输出：每个 base 输出一个 JSON：GFR_xxxxx.json
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

# ---- 仅用到的 OCC / occwl 依赖 ----
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import topods_Face
# ----

# ============ 你给的四种标签配置（也可从外部 JSON 读入） ============
DEFAULT_ATTRIBUTE_CONFIG = {
    "lables": {
        "clip": 1,
        "boss": 2,
        "rib":  3,
        "contact": 4
    }
}
# ===============================================================


# ----------------- 几何点提取与面索引器（与原脚本一致思路） -----------------
class BrepFeaturePointAnalyzer:
    """
    用于：
      - 从 STEP 读 shape
      - 为每个面分配递增 face_id（0..N-1）
      - 为每个顶点量化后建立 顶点ID -> 面ID集合 的映射
    """
    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance
        self.point_id_to_coord = {}        # 点ID -> 坐标
        self.point_id_to_faces = defaultdict(set)  # 点ID -> {face_id,...}
        self.face_id_to_points = defaultdict(set)  # face_id -> {点ID,...}
        self.face_id_to_actual_face = {}   # face_id -> TopoDS_Face
        self.next_face_id = 0

    def _quantize_point(self, p) -> str:
        # 注意：这里用一位小数（与你提供脚本中一致，避免不必要的几何偏差）
        return f"{p.X():.1f}_{p.Y():.1f}_{p.Z():.1f}"

    def extract_face_feature_points(self, face):
        pts = []
        exp = TopExp_Explorer(face, TopAbs_VERTEX)
        while exp.More():
            v = exp.Current()
            pts.append(BRep_Tool.Pnt(v))
            exp.Next()
        return pts

    def extract_shape_feature_points(self, shape):
        pts = []
        exp = TopExp_Explorer(shape, TopAbs_VERTEX)
        while exp.More():
            v = exp.Current()
            pts.append(BRep_Tool.Pnt(v))
            exp.Next()
        return pts

    def load_brep_from_step(self, file_path: str):
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(file_path))
        if status != 1:  # IFSelect_RetDone == 1
            raise RuntimeError(f"STEP read failed: {file_path}")
        reader.TransferRoot()
        shape = reader.Shape()

        # 先收集全局点（用于交集快速判断）
        for p in self.extract_shape_feature_points(shape):
            pid = self._quantize_point(p)
            if pid not in self.point_id_to_coord:
                self.point_id_to_coord[pid] = p

        # 面遍历与点-面映射
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = topods_Face(exp.Current())
            fid = self.next_face_id
            self.next_face_id += 1
            self.face_id_to_actual_face[fid] = face

            for p in self.extract_face_feature_points(face):
                pid = self._quantize_point(p)
                self.point_id_to_faces[pid].add(fid)
                self.face_id_to_points[fid].add(pid)
            exp.Next()

    # 查询接口
    def find_shared_points_with(self, other):
        # 两模型共享的量化点集合
        return set(self.point_id_to_coord.keys()) & set(other.point_id_to_coord.keys())

    def faces_of_point(self, pid: str):
        return self.point_id_to_faces.get(pid, set())

    def points_of_face(self, fid: int):
        return self.face_id_to_points.get(fid, set())


def find_matching_faces(analyzer_base: BrepFeaturePointAnalyzer,
                        analyzer_feat: BrepFeaturePointAnalyzer):
    """
    基于共享顶点 -> 候选面对 -> 点集完全相同 判定匹配
    返回：在 base 中的 face_id 列表
    """
    matching = []
    shared_pts = analyzer_base.find_shared_points_with(analyzer_feat)

    cand_pairs = set()
    for pid in shared_pts:
        for f1 in analyzer_base.faces_of_point(pid):
            for f2 in analyzer_feat.faces_of_point(pid):
                cand_pairs.add((f1, f2))

    for f1, f2 in cand_pairs:
        if analyzer_base.points_of_face(f1) == analyzer_feat.points_of_face(f2):
            matching.append(f1)

    return matching
# ---------------------------------------------------------------------


# --------------------- 目录扫描与配对 ---------------------
BASE_RE = re.compile(r"^(GFR_\d{5})\.step$", re.IGNORECASE)
CHILD_RE = re.compile(r"^(GFR_\d{5})_([A-Za-z0-9]+)\.step$", re.IGNORECASE)

def scan_step_dir(step_dir: str):
    """
    扫描目录，返回：
      bases: { 'GFR_00001': 'GFR_00001.step', ... }
      children: { 'GFR_00001': { 'clip': 'GFR_00001_clip.step', 'boss': ... }, ... }
    """
    step_dir = Path(step_dir)
    bases = {}
    children = defaultdict(dict)

    for f in step_dir.iterdir():
        if not f.is_file() or f.suffix.lower() != ".step":
            continue
        stem = f.stem

        m_base = BASE_RE.match(f.name)
        if m_base:
            key = m_base.group(1)  # 'GFR_00001'
            bases[key] = f.name
            continue

        m_child = CHILD_RE.match(f.name)
        if m_child:
            key = m_child.group(1)        # 'GFR_00001'
            feat = m_child.group(2).lower()  # 'clip' / 'boss' / 'rib' / 'contact' / etc
            children[key][feat] = f.name

    return bases, children
# --------------------------------------------------------


# --------------------- 主流程：为每个 base 生成 JSON ---------------------
def build_labels_for_base(step_dir: str,
                          base_filename: str,
                          child_map_for_base: dict,
                          attribute_config: dict):
    """
    为某个 base 计算 labels：
      - 未匹配面置 0
      - 对存在的子特征（且在 attribute_config 中）逐个打标
    返回：label_dict（key: str(face_id), value: int）
    """
    labels_cfg = {k.lower(): int(v) for k, v in attribute_config.get("lables", {}).items()}

    base_path = Path(step_dir) / base_filename
    base_an = BrepFeaturePointAnalyzer()
    base_an.load_brep_from_step(str(base_path))

    # 初始化：所有面都为 0
    label_dict = {str(fid): 0 for fid in range(base_an.next_face_id)}

    # 对每个实际存在的子特征文件匹配并标注
    for feat_name, child_filename in child_map_for_base.items():
        feat_key = feat_name.lower()
        if feat_key not in labels_cfg:
            # 非配置内特征，跳过
            continue
        child_path = Path(step_dir) / child_filename

        feat_an = BrepFeaturePointAnalyzer()
        feat_an.load_brep_from_step(str(child_path))

        matched = find_matching_faces(base_an, feat_an)
        for fid in matched:
            label_dict[str(fid)] = labels_cfg[feat_key]

    return label_dict


def generate_labels(step_dir: str,
                    out_dir: str,
                    attribute_config: dict = None):
    """
    step_dir: 输入 STEP 目录
    out_dir : 输出 JSON 目录（不存在则创建）
    attribute_config: 包含 "lables" 字段的 dict；若 None 则用 DEFAULT_ATTRIBUTE_CONFIG
    """
    attribute_config = attribute_config or DEFAULT_ATTRIBUTE_CONFIG
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bases, children = scan_step_dir(step_dir)

    for base_key, base_file in bases.items():
        # 找这个 base 实际拥有的子特征（只用配置中支持的四种）
        child_map = children.get(base_key, {})
        # 仅挑选配置中存在的子特征，且文件实际存在
        filtered_child_map = {
            feat: fname
            for feat, fname in child_map.items()
            if feat.lower() in attribute_config.get("lables", {})
        }

        # 生成标签
        label_dict = build_labels_for_base(step_dir, base_file, filtered_child_map, attribute_config)

        # 写 JSON
        out_path = out_dir / f"{base_key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(label_dict, f, indent=4, ensure_ascii=False)

        print(f"[OK] {base_key}: {len(label_dict)} faces -> {out_path}")


# --------------------- CLI ---------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-generate labels for base STEP models using existing sub-feature STEP files.")
    parser.add_argument("--step_dir", default= "/data_hdd/dev01/dyk/dyk_data/GFR_dataset_0.95/step", help="包含 GFR_xxxxx.step 及 GFR_xxxxx_<feat>.step 的目录")
    parser.add_argument("--out_dir", default= "/data_hdd/dev01/dyk/dyk_data/GFR_dataset_label", help="输出 JSON 目录")
    parser.add_argument("--labels_json", help="可选：外部标签配置 JSON 路径，格式同 DEFAULT_ATTRIBUTE_CONFIG")

    args = parser.parse_args()

    if args.labels_json and os.path.isfile(args.labels_json):
        with open(args.labels_json, "r", encoding="utf-8") as f:
            attr_cfg = json.load(f)
    else:
        attr_cfg = DEFAULT_ATTRIBUTE_CONFIG

    generate_labels(args.step_dir, args.out_dir, attr_cfg)
