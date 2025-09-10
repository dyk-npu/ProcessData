import os
import pandas as pd
import re

# --- 配置区 ---
# 1. 存放 .step 文件的源文件夹路径 (您提到有3027个基础文件的那个文件夹)
step_folder_path = r"D:\CAD数据集\项目\GFR_Dataset_Final" # <--- 请务必修改为您的实际文件夹路径

# 2. CSV文件的完整路径
csv_file_path = r"C:\Users\20268\Desktop\项目\数据集\stratege_hybrid_point_1.0\hybrid_point_1.0.csv"    # <--- 请务必修改为您的实际CSV文件路径

# 3. CSV中包含文件名的列名 (根据您的截图，列A的标题是'文件名')
column_name = "文件名"
# --- 配置区结束 ---


def find_missing_files_in_csv(folder_path, csv_path, col_name):
    """
    比较文件夹中的基础文件名(仅限GFR_xxxxx.step)和CSV文件中的文件名列表，
    找出在文件夹中作为基础文件存在，但在CSV中缺失的文件。
    """
    print("--- 开始查找缺失文件 ---")

    # --- 步骤 1: 从文件夹中仅获取 'GFR_xxxxx.step' 格式的文件 ---
    if not os.path.isdir(folder_path):
        print(f"错误：找不到指定的文件夹路径: {folder_path}")
        return

    folder_base_names = set()
    # 这个正则表达式现在非常严格：
    # ^ 表示字符串的开始
    # GFR_\d{5} 匹配 "GFR_" 加上5个数字
    # \.step$ 匹配 ".step" 并且是字符串的结尾
    # 这样就能精确地只匹配 GFR_01234.step 而排除 GFR_01234_boss.step
    pattern = re.compile(r'^GFR_\d{5}\.step$')

    for filename in os.listdir(folder_path):
        if pattern.match(filename):
            # 从文件名 "GFR_01234.step" 中提取出 "GFR_01234"
            base_name = filename[:-5] # 移除最后5个字符 ".step"
            folder_base_names.add(base_name)

    if not folder_base_names:
        print(f"警告：在文件夹 '{folder_path}' 中没有找到任何符合 'GFR_XXXXX.step' 格式的基础文件。")
        return

    print(f"在文件夹中找到 {len(folder_base_names)} 个 'GFR_xxxxx.step' 格式的基础文件。")

    # --- 步骤 2: 从CSV文件中获取所有唯一的文件名 (此部分逻辑不变) ---
    try:
        df = pd.read_csv(csv_path, usecols=[col_name])
        csv_base_names = set(df[col_name].dropna().unique())
        print(f"在CSV文件的 '{col_name}' 列中找到 {len(csv_base_names)} 个唯一的文件名。")
    except FileNotFoundError:
        print(f"错误：找不到CSV文件: {csv_path}")
        return
    except KeyError:
        print(f"错误：CSV文件中找不到名为 '{col_name}' 的列。请检查列名是否正确。")
        return
    except Exception as e:
        print(f"读取CSV文件时发生错误: {e}")
        return

    # --- 步骤 3: 找出差集 (存在于文件夹列表，但不存在于CSV列表) ---
    missing_files = sorted(list(folder_base_names - csv_base_names))

    # --- 步骤 4: 打印结果 ---
    print("\n" + "="*50)
    print(" " * 19 + "查找结果")
    print("="*50)
    
    if not missing_files:
        print("\n恭喜！文件夹中所有的基础文件(GFR_xxxxx.step)都在CSV中被找到。")
    else:
        print(f"\n找到了 {len(missing_files)} 个在文件夹中作为基础文件存在、但在CSV中缺失的文件：")
        for file in missing_files:
            print(f"  - {file}")

    print("\n" + "="*50)


# --- 运行脚本 ---
if __name__ == "__main__":
    if "C:\\path\\to\\" in step_folder_path or "C:\\path\\to\\" in csv_file_path:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 请先在脚本顶部的配置区修改 'step_folder_path' 和 'csv_file_path' !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        find_missing_files_in_csv(
            folder_path=step_folder_path,
            csv_path=csv_file_path,
            col_name=column_name
        )