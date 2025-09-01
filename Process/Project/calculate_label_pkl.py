import os
import pickle
import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE

def load_labels_from_pkl(label_filename):
    """
    从 .pkl 文件中加载标签数据。
    """
    try:
        with open(label_filename, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'face_labels' in data:
            return data['face_labels']
        else:
            print(f"警告: .pkl 文件 {os.path.basename(label_filename)} 的格式不正确或缺少 'face_labels' 键。")
            return None
    except Exception as e:
        print(f"错误: 无法加载 .pkl 文件 {label_filename}。错误: {e}")
        return None

def count_faces_in_step_file(step_filename):
    """
    计算一个 .step 文件中的面的数量。
    """
    if not os.path.exists(step_filename):
        return -1 

    reader = STEPControl_Reader()
    status = reader.ReadFile(step_filename)

    if status != 1:
        print(f"警告: 无法读取 STEP 文件: {step_filename}")
        return -1

    reader.TransferRoots()
    shape = reader.OneShape()
    
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_count = 0
    while face_explorer.More():
        face_count += 1
        face_explorer.Next()
        
    return face_count

def verify_feature_labels(step_directory, pkl_directory):
    """
    在指定目录中验证特征标签的准确性，并生成包含所有信息的日志。
    """
    discrepancies = []
    skipped_files_log = []
    feature_mapping = {'clip': 1, 'boss': 2, 'rib': 3, 'contact': 4}

    # ---- 初始化所有计数器 ----
    # 宏观计数器
    total_files_scanned = 0
    total_pkl_labels = 0
    total_step_faces_in_labeled_files = 0
    
    # 新增：分特征类型的聚合计数器
    feature_stats = {
        name: {'pkl_count': 0, 'step_face_count': 0} for name in feature_mapping.keys()
    }

    # ---- 遍历并处理文件 ----
    for filename in os.listdir(step_directory):
        if filename.startswith("GFR_") and filename.endswith(".step") and "_" not in os.path.splitext(filename)[0].split("GFR_")[1]:
            total_files_scanned += 1
            base_name = os.path.splitext(filename)[0]
            
            pkl_file_path = os.path.join(pkl_directory, f"{base_name}.pkl")

            if not os.path.exists(pkl_file_path):
                skip_message = f"文件 {base_name}: 在 '{os.path.basename(pkl_directory)}' 目录中找不到对应的 .pkl 文件。"
                print(f"信息: {skip_message}")
                skipped_files_log.append(skip_message)
                continue

            print(f"正在处理: {base_name}")
            labels = load_labels_from_pkl(pkl_file_path)
            
            if labels is None:
                continue

            # 累加宏观数据
            total_pkl_labels += len(labels)
            main_step_file_path = os.path.join(step_directory, filename)
            face_count = count_faces_in_step_file(main_step_file_path)
            if face_count != -1:
                total_step_faces_in_labeled_files += face_count

            # 计算当前文件的分特征标签数
            label_counts = {
                name: sum(1 for label in labels.values() if int(label) == label_id)
                for name, label_id in feature_mapping.items()
            }

            # 累加分特征数据并进行微观比对
            for feature_name, label_id in feature_mapping.items():
                # 累加 PKL 标签数到分特征聚合计数器
                feature_stats[feature_name]['pkl_count'] += label_counts[feature_name]
                
                sub_step_file_path = os.path.join(step_directory, f"{base_name}_{feature_name}.step")
                
                if os.path.exists(sub_step_file_path):
                    step_face_count = count_faces_in_step_file(sub_step_file_path)
                    
                    if step_face_count != -1:
                        # 累加 STEP 面数到分特征聚合计数器
                        feature_stats[feature_name]['step_face_count'] += step_face_count
                        
                        # 微观比对 (单个文件)
                        if step_face_count != label_counts[feature_name]:
                            mismatch_message = (f"文件 {base_name} 中的特征 '{feature_name}' 存在不匹配: "
                                                f"STEP 文件中的面数量 = {step_face_count}, "
                                                f"PKL 文件中的标签数量 = {label_counts[feature_name]}")
                            print(mismatch_message)
                            discrepancies.append(mismatch_message)

    # ---- 生成日志和最终输出 ----
    log_file_path = os.path.join(step_directory, "label_verification_log.txt")

    # 计算整体覆盖率
    coverage_rate_str = "N/A"
    if total_step_faces_in_labeled_files > 0:
        coverage_rate = (total_pkl_labels / total_step_faces_in_labeled_files)
        coverage_rate_str = f"{coverage_rate:.2%}"
        
    # 创建摘要头
    summary_header = f"""
==================================================
              标签验证结果摘要
==================================================

[宏观覆盖率统计]
总计标签数量 (来自所有有效PKL文件): {total_pkl_labels}
总计主STEP文件面数量 (仅限有对应PKL的文件): {total_step_faces_in_labeled_files}
整体标签覆盖率 (总标签数 / 总面数): {coverage_rate_str}

--------------------------------------------------

[分特征类型聚合准确度]"""

    # 计算并格式化分特征准确度
    feature_accuracy_lines = []
    for name, stats in feature_stats.items():
        pkl_total = stats['pkl_count']
        step_total = stats['step_face_count']
        accuracy_str = "N/A"
        if step_total > 0:
            accuracy = (pkl_total / step_total)
            accuracy_str = f"{accuracy:.2%}"
        
        line = f"{name.capitalize():<8} 特征: PKL总数={pkl_total:<5}, STEP总面数={step_total:<5}, 准确度={accuracy_str}"
        feature_accuracy_lines.append(line)

    summary_footer = f"""
--------------------------------------------------

[文件处理与不匹配项统计]
总共扫描的主 STEP 文件数量: {total_files_scanned}
因缺少对应 .pkl 文件而跳过的数量: {len(skipped_files_log)}
发现特征数量不匹配的项总数: {len(discrepancies)}

==================================================
"""
    
    # 拼接完整的报告
    full_report = summary_header + "\n" + "\n".join(feature_accuracy_lines) + summary_footer

    # 写入文件并打印到控制台
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(full_report.strip() + "\n")
            
            if discrepancies:
                f.write("\n\n[不匹配项详情]\n" + "-"*40 + "\n")
                f.write("\n".join(discrepancies))
            else:
                f.write("\n\n恭喜！未在可验证的文件中发现任何特征数量不一致之处。\n")

            if skipped_files_log:
                f.write("\n\n[因缺少 .pkl 文件而被跳过的文件]\n" + "-"*40 + "\n")
                f.write("\n".join(skipped_files_log))
        
        print(full_report)
        print(f"验证完成。所有详情已记录在日志文件中: {log_file_path}")

    except Exception as e:
        print(f"\n错误: 无法写入日志文件 {log_file_path}。错误: {e}")

# --- 使用说明 ---
if __name__ == "__main__":
    # **重要**: 请在此处设置您的数据目录路径
    step_directory_path = "D:\CAD数据集\项目\GFR_Dataset"  
    pkl_directory_path = "D:\CAD数据集\项目\GFR_TrainingData_Modify"

    if not os.path.isdir(step_directory_path):
        print(f"错误: STEP 文件目录 '{step_directory_path}' 不存在。请更新路径。")
    elif not os.path.isdir(pkl_directory_path):
        print(f"错误: PKL 文件目录 '{pkl_directory_path}' 不存在。请更新路径。")
    else:
        verify_feature_labels(step_directory_path, pkl_directory_path)



