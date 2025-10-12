import os
import shutil

# --- 请在这里修改为您要整理的文件夹路径 ---
target_directory = "E:\CAD数据集\SolidLetters\step"
# -----------------------------------------

# 检查指定的路径是否存在
if not os.path.isdir(target_directory):
    print(f"错误：找不到目录 '{target_directory}'。请检查路径是否正确。")
else:
    # 定义要创建的文件夹名称
    step_folder = os.path.join(target_directory, 'step_files')
    meta_folder = os.path.join(target_directory, 'meta_files')

    # 在目标目录下创建这两个文件夹（如果它们还不存在）
    if not os.path.exists(step_folder):
        os.makedirs(step_folder)
        print(f"文件夹 '{step_folder}' 已创建。")

    if not os.path.exists(meta_folder):
        os.makedirs(meta_folder)
        print(f"文件夹 '{meta_folder}' 已创建。")

    # 遍历目标目录下的所有文件
    for filename in os.listdir(target_directory):
        source_path = os.path.join(target_directory, filename)

        # 确保我们处理的是文件而不是文件夹
        if os.path.isfile(source_path):
            # 检查文件是否是 .step 文件
            if filename.endswith('.step'):
                destination_path = os.path.join(step_folder, filename)
                shutil.move(source_path, destination_path)
                print(f"已将 {filename} 移动到 {step_folder}")

            # 检查文件是否是 .meta 文件
            elif filename.endswith('.meta'):
                destination_path = os.path.join(meta_folder, filename)
                shutil.move(source_path, destination_path)
                print(f"已将 {filename} 移动到 {meta_folder}")

    print("\n文件整理完成！")