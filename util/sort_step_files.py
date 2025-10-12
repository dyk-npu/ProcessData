import os
import shutil

# --- 请在这里修改为您要整理的 "step_files" 文件夹的路径 ---
step_directory = "E:\CAD数据集\SolidLetters\step\step_files"
# -----------------------------------------------------------

# 检查指定的 "step" 目录是否存在
if not os.path.isdir(step_directory):
    print(f"错误：找不到目录 '{step_directory}'。请检查路径是否正确。")
else:
    # 在 "step" 目录内定义 "upper" 和 "lower" 文件夹的完整路径
    upper_folder = os.path.join(step_directory, 'upper')
    lower_folder = os.path.join(step_directory, 'lower')

    # 创建这两个文件夹（如果它们还不存在）
    if not os.path.exists(upper_folder):
        os.makedirs(upper_folder)
        print(f"文件夹 '{upper_folder}' 已创建。")

    if not os.path.exists(lower_folder):
        os.makedirs(lower_folder)
        print(f"文件夹 '{lower_folder}' 已创建。")

    # 遍历 "step" 目录下的所有文件和文件夹
    for filename in os.listdir(step_directory):
        source_path = os.path.join(step_directory, filename)

        # 确保我们只处理文件，跳过已经创建的 "upper" 和 "lower" 文件夹
        if os.path.isfile(source_path):
            # 检查文件名是否包含 "_lower" (不区分大小写)
            if '_lower' in filename.lower():
                destination_path = os.path.join(lower_folder, filename)
                shutil.move(source_path, destination_path)
                print(f"已将 {filename} 移动到 'lower' 文件夹")

            # 检查文件名是否包含 "_upper" (不区分大小写)
            elif '_upper' in filename.lower():
                destination_path = os.path.join(upper_folder, filename)
                shutil.move(source_path, destination_path)
                print(f"已将 {filename} 移动到 'upper' 文件夹")

    print("\n'step' 文件夹内文件分类完成！")