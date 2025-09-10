import pandas as pd

# CSV 文件路径
csv_file_path = r"C:\Users\20268\Desktop\项目\数据集\stratege_point\accuracy_point.csv"

def count_valid_files(csv_path):
    try:
        # 读取 CSV 文件
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # 清理空格
        df['文件名'] = df['文件名'].astype(str).str.strip()
        df['特征'] = df['特征'].astype(str).str.strip()

        # 处理“准确率”列：去掉百分号，转为浮点数
        df['准确率'] = df['准确率'].astype(str).str.replace('%', '', regex=False).astype(float) / 100.0

    except Exception as e:
        print(f"读取或预处理 CSV 出错: {e}")
        return

    # 按“文件名”分组
    grouped = df.groupby('文件名')

    # 计数器
    valid_count = 0
    valid_files = []

    for file_name, group in grouped:
        # 判断该文件所有子特征的准确率是否都在 (0.95, 1] 之间
        if ((group['准确率'] > 0.95) & (group['准确率'] <= 1)).all():
            valid_count += 1
            valid_files.append(file_name)

    print(f"\n符合条件的文件总数: {valid_count}")


    return valid_count, valid_files


# --- 运行 ---
if __name__ == "__main__":
    count_valid_files(csv_file_path)