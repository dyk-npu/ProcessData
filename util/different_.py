import pandas as pd

# CSV 文件路径
csv_file_path = r"C:\Users\20268\Desktop\项目\数据集\stratege_point\accuracy_point.csv"

def count_unique_filenames(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # 清理空格
        df['文件名'] = df['文件名'].astype(str).str.strip()

        # 计算唯一文件名数量
        unique_files = df['文件名'].nunique()

        print(f"文件名列中共有 {unique_files} 个不同的文件名。")

        return unique_files

    except Exception as e:
        print(f"读取或处理 CSV 出错: {e}")

# --- 运行 ---
if __name__ == "__main__":
    count_unique_filenames(csv_file_path)
