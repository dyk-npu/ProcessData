import os
import re
from tqdm import tqdm  # 导入 tqdm 库


def rename_and_move_files(folder_path, new_folder_name="label2", prefix="single_"):
    """
    按照文件名中的数字重新排序，重命名文件，并将它们移动到新的文件夹。
    新文件夹中的文件编号从 0 开始。
    """
    # 创建新文件夹路径
    new_folder_path = os.path.join(folder_path, new_folder_name)

    # 如果新文件夹不存在，则创建
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"Created new folder: {new_folder_path}")

    # 获取文件夹中的所有文件名
    all_files = os.listdir(folder_path)

    # 筛选出以指定前缀开头且以 .step 结尾的文件
    json_files = [f for f in all_files if f.startswith(prefix) and f.endswith(".json")]

    # 提取文件名中的数字，并按数字排序
    file_data = []
    for filename in json_files:
        match = re.search(rf"{prefix}(\d+)\.json", filename)  # 匹配数字部分
        if match:
            number = int(match.group(1))  # 提取数字
            file_data.append((number, filename))

    # 按数字排序
    file_data.sort(key=lambda x: x[0])

    # 按新顺序重命名文件并移动到新文件夹
    for new_index, (old_number, old_filename) in enumerate(tqdm(file_data, desc="Processing Files"), start=7989):
        # 构造新的文件名
        new_filename = f"{prefix}{new_index}.json"

        # 构造完整路径
        old_file_path = os.path.join(folder_path, old_filename)
        new_file_path = os.path.join(new_folder_path, new_filename)

        # 移动并重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Moved and renamed: {old_filename} -> {new_filename}")


# 使用示例
if __name__ == "__main__":
    folder_path = "C:\\Users\\20268\Desktop\Project\ProcessData\Data\CBF\\addData\label"  # 替换为你的文件夹路径
    rename_and_move_files(folder_path)
