import os
import re

def find_missing_files(folder_path):
    """
    统计文件夹中 GFR_xxxxx.step 基础文件的数量，并找出不连续的文件。
    """
    base_files = []
    # 匹配模式：严格 GFR_数字.step
    pattern = re.compile(r"^GFR_(\d+)\.step$")
    numbers = []

    print(f"正在扫描文件夹: {folder_path}")

    # 1. 遍历文件，提取所有基础文件的编号
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            # match.group(0) 是整个文件名，match.group(1) 是第一个括号里的内容，即数字
            numbers.append(int(match.group(1)))
            base_files.append(filename)

    if not numbers:
        print("未找到任何 GFR_xxxxx.step 格式的基础文件。")
        return

    # 2. 打印基础文件信息
    print("-" * 30)
    print(f"基础文件总数: {len(base_files)}")
    # 如果需要，可以取消下面的注释来查看所有基础文件
    # print("基础文件列表:")
    # for f in sorted(base_files):
    #     print("  " + f)
    print("-" * 30)


    # 3. 找出不连续的文件
    numbers.sort() # 对数字进行排序
    
    min_num = numbers[0]
    max_num = numbers[-1]
    
    # 生成一个从最小到最大编号的完整序列
    full_sequence = set(range(min_num, max_num + 1))
    
    # 与现有文件编号进行对比，找出差集
    missing_numbers = sorted(list(full_sequence - set(numbers)))

    if missing_numbers:
        print(f"发现 {len(missing_numbers)} 个不连续的文件（缺失的编号）:")
        for num in missing_numbers:
            # 格式化输出，例如：GFR_1236.step
            print(f"  - GFR_{num}.step")
    else:
        print("文件序列是连续的，没有发现缺失的文件。")
    print("-" * 30)


if __name__ == "__main__":
    # --- 重要：请将这里的路径修改为你的实际文件夹路径 ---
    folder = "D:\CAD数据集\项目\GFR_Dataset_Final"
    
    # 检查路径是否存在
    if os.path.isdir(folder):
        find_missing_files(folder)
    else:
        print(f"错误：文件夹路径不存在 -> {folder}")
        print("请创建一个名为 'test_folder' 的文件夹并放入一些文件用于测试。")
        # --- 为了方便演示，我们创建一个临时文件夹并生成一些测试文件 ---
        test_folder = "test_folder"
        print(f"正在创建临时文件夹 '{test_folder}' 用于演示...")
        os.makedirs(test_folder, exist_ok=True)
        test_files = [
            "GFR_1234.step",
            "GFR_1235.step",
            "GFR_1235_clip.step", # 会被忽略
            "GFR_1237.step",
            "GFR_1238.step",
            "GFR_1241.step",
            "random_file.txt" # 会被忽略
        ]
        for f in test_files:
            open(os.path.join(test_folder, f), 'a').close()
        
        print("\n--- 开始演示 ---")
        find_missing_files(test_folder)