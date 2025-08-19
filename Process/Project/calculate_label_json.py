import os
import json  # 删除了 pickle，引入了 json
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE

"""
在指定目录中验证特征标签的准确性，并生成包含所有信息的日志。
日志现在是实时写入的。
此版本从 .json 文件加载标签。
"""

def load_labels_from_json(label_filename):
    """
    从 .json 文件中加载标签数据。
    """
    try:
        with open(label_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 假设JSON文件的根就是一个字典，如示例所示
        if isinstance(data, dict):
            return data
        else:
            print(f"警告: .json 文件 {os.path.basename(label_filename)} 的格式不正确，根元素不是一个字典。")
            return None
    except json.JSONDecodeError as e:
        print(f"错误: 解析 .json 文件 {label_filename} 失败。错误: {e}")
        return None
    except Exception as e:
        print(f"错误: 无法加载 .json 文件 {label_filename}。错误: {e}")
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

def verify_feature_labels(step_directory, json_directory):
    """
    在指定目录中验证特征标签的准确性，并生成包含所有信息的日志。
    日志现在是实时写入的。
    """
    log_file_path = os.path.join("/data_hdd/dev01/dyk/dyk_data/log", "label_verification_log.txt")
    discrepancies = []
    skipped_files_log = []
    feature_mapping = {'clip': 1, 'boss': 2, 'rib': 3, 'contact': 4}

    # ---- 初始化所有计数器 ----
    total_files_scanned = 0
    total_json_labels = 0
    total_step_faces_in_labeled_files = 0
    
    feature_stats = {
        name: {'json_count': 0, 'step_face_count': 0} for name in feature_mapping.keys()
    }

    try:
        # 在处理开始前打开日志文件，并保持打开状态
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write("==================================================\n")
            log_file.write("              实时处理日志\n")
            log_file.write("==================================================\n\n")
            log_file.flush()

            # ---- 遍历并处理文件 ----
            for filename in os.listdir(step_directory):
                if filename.startswith("GFR_") and filename.endswith(".step") and "_" not in os.path.splitext(filename)[0].split("GFR_")[1]:
                    total_files_scanned += 1
                    base_name = os.path.splitext(filename)[0]
                    
                    processing_message = f"正在处理: {base_name}"
                    print(processing_message)
                    log_file.write(processing_message + "\n")
                    log_file.flush()
                    
                    json_file_path = os.path.join(json_directory, f"{base_name}.json")

                    if not os.path.exists(json_file_path):
                        skip_message = f"文件 {base_name}: 在 '{os.path.basename(json_directory)}' 目录中找不到对应的 .json 文件。"
                        print(f"信息: {skip_message}")
                        log_file.write(f"-> 跳过: {skip_message}\n")
                        log_file.flush()
                        skipped_files_log.append(skip_message)
                        continue

                    labels = load_labels_from_json(json_file_path)
                    
                    if labels is None:
                        # 记录加载失败的信息
                        load_fail_message = f"-> 失败: 无法从 {base_name}.json 加载标签。"
                        print(load_fail_message)
                        log_file.write(load_fail_message + "\n")
                        log_file.flush()
                        continue

                    total_json_labels += len(labels)
                    main_step_file_path = os.path.join(step_directory, filename)
                    face_count = count_faces_in_step_file(main_step_file_path)
                    if face_count != -1:
                        total_step_faces_in_labeled_files += face_count

                    label_counts = {
                        name: sum(1 for label in labels.values() if int(label) == label_id)
                        for name, label_id in feature_mapping.items()
                    }

                    for feature_name, label_id in feature_mapping.items():
                        feature_stats[feature_name]['json_count'] += label_counts[feature_name]
                        
                        sub_step_file_path = os.path.join(step_directory, f"{base_name}_{feature_name}.step")
                        
                        if os.path.exists(sub_step_file_path):
                            step_face_count = count_faces_in_step_file(sub_step_file_path)
                            
                            if step_face_count != -1:
                                feature_stats[feature_name]['step_face_count'] += step_face_count
                                
                                if step_face_count != label_counts[feature_name]:
                                    mismatch_message = (f"文件 {base_name} 中的特征 '{feature_name}' 存在不匹配: "
                                                        f"STEP 文件中的面数量 = {step_face_count}, "
                                                        f"JSON 文件中的标签数量 = {label_counts[feature_name]}")
                                    print(mismatch_message)
                                    # 实时写入不匹配信息
                                    log_file.write(f"-> 不匹配: {mismatch_message}\n")
                                    log_file.flush()
                                    discrepancies.append(mismatch_message)

            # ---- 所有文件处理完毕，开始生成并写入最终的摘要报告 ----
            
            # 计算整体覆盖率
            coverage_rate_str = "N/A"
            if total_step_faces_in_labeled_files > 0:
                coverage_rate = (total_json_labels / total_step_faces_in_labeled_files)
                coverage_rate_str = f"{coverage_rate:.2%}"
                
            # 创建摘要头
            summary_header = f"""
==================================================
              标签验证结果摘要
==================================================

[宏观覆盖率统计]
总计标签数量 (来自所有有效JSON文件): {total_json_labels}
总计主STEP文件面数量 (仅限有对应JSON的文件): {total_step_faces_in_labeled_files}
整体标签覆盖率 (总标签数 / 总面数): {coverage_rate_str}

--------------------------------------------------

[分特征类型聚合准确度]"""

            # 计算并格式化分特征准确度
            feature_accuracy_lines = []
            for name, stats in feature_stats.items():
                json_total = stats['json_count']
                step_total = stats['step_face_count']
                accuracy_str = "N/A"
                if step_total > 0:
                    accuracy = (json_total / step_total)
                    accuracy_str = f"{accuracy:.2%}"
                
                line = f"{name.capitalize():<8} 特征: JSON总数={json_total:<5}, STEP总面数={step_total:<5}, 准确度={accuracy_str}"
                feature_accuracy_lines.append(line)

            summary_footer = f"""
--------------------------------------------------

[文件处理与不匹配项统计]
总共扫描的主 STEP 文件数量: {total_files_scanned}
因缺少对应 .json 文件而跳过的数量: {len(skipped_files_log)}
发现特征数量不匹配的项总数: {len(discrepancies)}

==================================================
"""
            
            # 拼接完整的报告
            full_report = summary_header + "\n" + "\n".join(feature_accuracy_lines) + summary_footer

            # 将最终报告写入日志文件末尾并打印到控制台
            log_file.write("\n\n" + full_report.strip() + "\n")
            
            if discrepancies:
                log_file.write("\n\n[不匹配项详情]\n" + "-"*40 + "\n")
                log_file.write("\n".join(discrepancies))
            else:
                log_file.write("\n\n恭喜！未在可验证的文件中发现任何特征数量不一致之处。\n")

            if skipped_files_log:
                log_file.write("\n\n[因缺少 .json 文件而被跳过的文件]\n" + "-"*40 + "\n")
                log_file.write("\n".join(skipped_files_log))
        
        # 脚本运行结束后，在控制台打印最终报告
        print("\n" + "="*50)
        print("              最终摘要报告")
        print("="*50)
        print(full_report)
        print(f"验证完成。所有详情已记录在日志文件中: {log_file_path}")

    except Exception as e:
        print(f"\n错误: 无法写入日志文件 {log_file_path}。错误: {e}")

# --- 使用说明 ---
if __name__ == "__main__":
    # **重要**: 请在此处设置您的数据目录路径
    step_directory_path = "/data_hdd/dataset/GFR_Dataset_Final"  
    json_directory_path = "/data_hdd/dev01/dyk/dyk_data/GFR_dataset_label_my" # <--- 目录变量名已更新

    if not os.path.isdir(step_directory_path):
        print(f"错误: STEP 文件目录 '{step_directory_path}' 不存在。请更新路径。")
    elif not os.path.isdir(json_directory_path):
        print(f"错误: JSON 文件目录 '{json_directory_path}' 不存在。请更新路径。") # <--- 错误信息已更新
    else:
        verify_feature_labels(step_directory_path, json_directory_path)