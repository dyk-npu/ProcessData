import pandas as pd

# --- 请在这里配置您的文件信息 ---

# 1. 输入您的原始 CSV 文件名
input_filename = 'C:\\Users\\20268\\Desktop\\项目\\数据集\\label_file_analysis.csv' 

# 2. 指定您希望输出的新 CSV 文件名
output_filename = 'C:\\Users\\20268\\Desktop\\项目\\数据集\\label_file_sorted2.csv'

# 3. 指定要排序的列名 (根据图片，应该是'准确率(%)')
column_to_sort_by = '准确率(%)'

# --- 脚本主程序 ---
try:
    # 步骤 1: 读取 CSV 文件
    # 脚本会首先尝试用'utf-8'编码打开，如果失败，则会尝试'gbk'，这能兼容大多数情况
    try:
        df = pd.read_csv(input_filename, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_filename, encoding='gbk')
    
    print(f"成功读取文件: '{input_filename}'")

    # 步骤 2: 检查排序列是否存在
    if column_to_sort_by not in df.columns:
        print(f"错误：在文件中找不到名为 '{column_to_sort_by}' 的列！")
        print(f"文件中的可用列有: {df.columns.tolist()}")
    else:
        # 步骤 3: 按照指定列进行升序排序 (从低到高)
        # ascending=True 表示升序
        sorted_df = df.sort_values(by=column_to_sort_by, ascending=True)
        print(f"已按 '{column_to_sort_by}' 列从低到高完成排序。")

        # 步骤 4: 将排序后的结果保存到新的 CSV 文件
        # index=False 表示不将 DataFrame 的索引写入文件
        # encoding='utf-8-sig' 确保在 Excel 中打开时中文不会显示为乱码
        sorted_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        
        print(f"排序结果已成功保存到新文件: '{output_filename}'")

except FileNotFoundError:
    print(f"错误：找不到文件 '{input_filename}'。请检查文件名是否正确，并确保文件和脚本在同一个文件夹下。")
except Exception as e:
    print(f"处理过程中发生了一个意料之外的错误: {e}")