import pandas as pd

# 定义要读取和保存的文件名
input_filename = r"C:\Users\20268\Desktop\项目\数据集\stratege_hybrid_point_1.0_attribute_1.0_V2.0\hybrid_point_1.0_attribute_1.0_V2.0.csv" # <--- 请将这里替换成您的原始文件名
output_filename = r"C:\Users\20268\Desktop\项目\数据集\stratege_hybrid_point_1.0_attribute_1.0_V2.0\acc_hybrid_point_1.0_attribute_1.0_V2.0.csv" # <--- 这是排序后新文件的名称

try:
    # 1. 读取CSV文件到pandas DataFrame
    df = pd.read_csv(input_filename, encoding='utf-8-sig')

    # 2. 数据清洗与转换
    # 创建一个新列用于排序，将“准确率”列中的百分号'%'去掉，并转换为数值
    df['sort_key'] = pd.to_numeric(df['准确率'].str.rstrip('%'), errors='coerce')

    # 3. 按“准确率”升序排序
    # ascending=True 表示从小到大（升序）排序
    sorted_df = df.sort_values(by='sort_key', ascending=True)

    # 4. 删除用于排序的辅助列
    sorted_df = sorted_df.drop(columns=['sort_key'])    

    # 5. 将排序后的结果保存到新的CSV文件
    sorted_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"成功！文件已根据“准确率”从小到大排序，并保存为 '{output_filename}'。")
    print("\n排序后数据预览 (准确率最低的几行)：")
    print(sorted_df.head()) # 打印排序后前5行以供预览

except FileNotFoundError:
    print(f"错误：找不到文件 '{input_filename}'。请确认文件名是否正确，以及文件是否与脚本在同一个目录下。")
except KeyError:
    print("错误：CSV文件中未找到名为“准确率”的列。请检查您的列名是否正确。")
except Exception as e:
    print(f"处理过程中发生未知错误: {e}")


