# import pandas as pd

# # --- 配置 ---
# # 请将 'your_csv_file.csv' 替换成您的原始CSV文件名
# input_filename = 'C:\\Users\\20268\\Desktop\\项目\\数据集\\生成训练数据过程中报错的样本列表.csv'
# # 这是排序后生成的新文件名，您可以按需修改
# output_filename = 'C:\\Users\\20268\\Desktop\\项目\\数据集\\sorted_error_file.csv'

# try:
#     # 读取CSV文件。根据您的截图，文件没有标题行，所以使用 header=None
#     df = pd.read_csv(input_filename, header=None)

#     # 获取用于排序的第一列的列名（在这里是 0）
#     sort_column_index = 0

#     # 创建一个临时的'sort_key'列，用于排序。
#     # 我们将第一列的字符串按'_'分割，取第二部分，并转换为整数。
#     df['sort_key'] = df[sort_column_index].str.split('_').str[1].astype(int)

#     # 根据'sort_key'列对整个表格进行排序
#     sorted_df = df.sort_values(by='sort_key')

#     # 删除临时的'sort_key'列，因为它在最终输出中是不需要的
#     sorted_df = sorted_df.drop(columns=['sort_key'])

#     # 将排序后的数据保存到新的CSV文件中
#     # index=False 表示不将行号写入文件
#     # header=False 表示不将列标题写入文件
#     sorted_df.to_csv(output_filename, index=False, header=False)

#     print(f"文件已成功排序并保存为: {output_filename}")

# except FileNotFoundError:
#     print(f"错误：找不到文件 '{input_filename}'。请检查文件名和路径是否正确。")
# except Exception as e:
#     print(f"处理过程中发生错误: {e}")

import pandas as pd
import os

# --- 配置 ---
# 请将 'your_file.csv' 替换成您的文件名（可以是 .csv 或 .xlsx）
input_filename = 'C:\\Users\\20268\\Desktop\\项目\\数据集\\生成训练数据过程中报错的样本列表.csv'
# 输出文件名会自动生成，无需修改

try:
    # --- 1. 读取文件 ---
    # 分离文件名和扩展名
    file_name, file_extension = os.path.splitext(input_filename)
    
    # 根据文件扩展名选择合适的读取函数
    if file_extension.lower() == '.csv':
        df = pd.read_csv(input_filename, header=None)
        output_filename = f"{file_name}_sorted.csv"
    elif file_extension.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_filename, header=None)
        output_filename = f"{file_name}_sorted.xlsx"
    else:
        # 如果文件格式不支持，则抛出错误
        raise ValueError("不支持的文件格式。请使用 .csv 或 .xlsx 文件。")

    # --- 2. 提取用于排序的数字 ---
    # 获取第一列的索引
    sort_column_index = 0
    
    # 创建一个临时列'sort_key'来存放提取的数字
    # 步骤分解:
    # 1. .str.split('_').str[1]  -> 从 "GFR_00283.step" 中得到 "00283.step"
    # 2. .str.split('.').str[0]   -> 从 "00283.step" 中得到 "00283"
    # 3. .astype(int)             -> 将字符串 "00283" 转换为整数 283
    df['sort_key'] = df[sort_column_index].str.split('_').str[1].str.split('.').str[0].astype(int)

    # --- 3. 排序 ---
    # 根据'sort_key'列对整个表格进行排序
    sorted_df = df.sort_values(by='sort_key')
    
    # 删除临时的'sort_key'列
    sorted_df = sorted_df.drop(columns=['sort_key'])

    # --- 4. 保存文件 ---
    # 根据原始文件类型保存排序后的文件
    if file_extension.lower() == '.csv':
        sorted_df.to_csv(output_filename, index=False, header=False)
    elif file_extension.lower() in ['.xlsx', '.xls']:
        sorted_df.to_excel(output_filename, index=False, header=False)

    print(f"文件已成功排序并保存为: {output_filename}")

except FileNotFoundError:
    print(f"错误：找不到文件 '{input_filename}'。请检查文件名和路径是否正确。")
except Exception as e:
    print(f"处理过程中发生错误: {e}")