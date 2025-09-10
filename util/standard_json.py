import json
import os
import zipfile
import argparse
from tqdm import tqdm

def format_json_dataset(zip_filepath, output_directory):
    """
    Reads a zip archive containing single-line JSON files, formats them,
    and saves them to a new directory.
    """
    # 步骤 1: 检查输入的zip文件是否存在
    if not os.path.exists(zip_filepath):
        print(f"错误: 找不到输入的zip文件 '{zip_filepath}'")
        return

    # 步骤 2: 创建输出文件夹（如果它还不存在）
    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"所有格式化后的文件将被保存在文件夹: '{output_directory}'")
    except OSError as e:
        print(f"错误: 创建文件夹 '{output_directory}' 失败: {e}")
        return
        
    # 步骤 3: 打开zip文件并开始处理
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            # 获取压缩包内所有文件的列表，用于进度条
            file_list = [member for member in zf.infolist() if not member.is_dir() and member.filename.endswith('.json')]
            print(f"在 {os.path.basename(zip_filepath)} 中找到了 {len(file_list)} 个JSON文件。")

            # 使用tqdm创建进度条，遍历并处理每个文件
            for member in tqdm(file_list, desc="正在格式化JSON文件"):
                try:
                    # 从zip中直接读取文件内容
                    with zf.open(member.filename) as file_in_zip:
                        # 读取单行字符串并解析
                        json_string = file_in_zip.read().decode('utf-8')
                        data = json.loads(json_string)

                    # 构建新文件的完整输出路径
                    output_path = os.path.join(output_directory, os.path.basename(member.filename))

                    # 将格式化后的内容写入新文件
                    with open(output_path, 'w', encoding='utf-8') as f_out:
                        json.dump(data, f_out, indent=4, ensure_ascii=False)

                except json.JSONDecodeError:
                    print(f"\n警告: 文件 '{member.filename}' 内容不是有效的JSON，已跳过。")
                except Exception as e:
                    print(f"\n处理文件 '{member.filename}' 时发生错误: {e}")

        print(f"\n处理完成！所有 {len(file_list)} 个文件都已格式化并保存。")

    except zipfile.BadZipFile:
        print(f"错误: '{zip_filepath}' 不是一个有效的zip文件。")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    # --- 如何运行 ---
    # 1. 安装tqdm库 (如果还没有的话):
    #    在你的终端或命令行里运行: pip install tqdm
    #
    # 2. 将此脚本保存为 .py 文件, 例如 'format_dataset.py'.
    #
    # 3. 将 'assemblies.zip' 文件和这个脚本放在同一个文件夹下。
    #
    # 4. 在终端或命令行里运行脚本，并提供两个参数：
    #    第一个是输入的zip文件名，第二个是输出的文件夹名。
    #
    #    示例命令:
    #    python format_dataset.py assemblies.zip formatted_assemblies
    # ------------------
    
    parser = argparse.ArgumentParser(
        description="格式化 AutoMate 数据集中的 'assemblies.zip' 文件，将其中的单行JSON转换为可读格式。"
    )
    parser.add_argument(
        "zip_filepath", 
        help="输入的 'assemblies.zip' 文件的路径。"
    )
    parser.add_argument(
        "output_directory", 
        help="用于保存格式化后JSON文件的新文件夹的名称。"
    )

    args = parser.parse_args()
    format_json_dataset(args.zip_filepath, args.output_directory)