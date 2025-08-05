import os


def find_missing_files(folder_path, start=8000, end=19999, prefix="double_", suffix=".step"):
    # 获取文件夹中的所有文件名
    files = os.listdir(folder_path)

    # 提取文件名中的编号部分
    existing_numbers = set()
    for file in files:
        if file.startswith(prefix) and file.endswith(suffix):
            try:
                # 提取编号部分并转换为整数
                number = int(file[len(prefix):-len(suffix)])  # 去掉前缀和后缀
                existing_numbers.add(number)
            except ValueError:
                # 如果文件名不符合预期格式，跳过
                print(f"跳过无效文件名: {file}")
                continue

    # 生成完整的编号范围
    full_range = set(range(start, end + 1))

    # 找出缺失的编号
    missing_numbers = full_range - existing_numbers

    # 返回排序后的缺失编号列表
    return sorted(missing_numbers)


# 使用示例
if __name__ == "__main__":
    folder_path = "C:\\Users\\20268\Desktop\Project\ProcessData\Data\CBF\data\step\step2"  # 替换为你的文件夹路径
    start = 8000
    end = 19999
    missing_files = find_missing_files(folder_path, start=start, end=end)

    if missing_files:
        print("缺少的文件编号：")
        for number in missing_files:
            print(f"double_{number}.step")
    else:
        print("没有缺少的文件。")