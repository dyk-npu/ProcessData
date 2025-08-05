import json
import os
from tqdm import tqdm  # 导入 tqdm 库


def sort_json_keys(input_file_path, output_file_path):
    """
    读取 JSON 文件，按键的数值排序，并将结果写入新的 JSON 文件。
    """
    # 读取 JSON 文件
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 按键的数值排序
    sorted_data = dict(sorted(data.items(), key=lambda item: int(item[0])))

    # 将排序后的数据写入新的 JSON 文件
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(sorted_data, file, indent=4, ensure_ascii=False)

    print(f"Sorted and saved: {output_file_path}")


def process_folder(input_folder, output_folder):
    """
    遍历文件夹中的所有 JSON 文件，对每个文件进行按键排序并保存。
    """
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # 获取文件夹中的所有 JSON 文件
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    # 使用 tqdm 显示进度条
    for json_file in tqdm(json_files, desc="Processing JSON Files", unit="file"):
        input_file_path = os.path.join(input_folder, json_file)
        output_file_path = os.path.join(output_folder, json_file)

        # 对每个 JSON 文件进行处理
        sort_json_keys(input_file_path, output_file_path)


# 使用示例
if __name__ == "__main__":
    input_folder = "C:\\Users\\20268\Desktop\Project\ProcessData\Data\CBF\data\label"  # 替换为你的输入文件夹路径
    output_folder = "C:\\Users\\20268\Desktop\Project\ProcessData\Data\CBF\data\label\label2"  # 替换为你的输出文件夹路径

    process_folder(input_folder, output_folder)