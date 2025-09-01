import pandas as pd

def find_unique_rows_in_a(file_a_path='A.csv', file_b_path='B.csv', key_columns=['文件名', '特征']):
    """
    比较两个CSV文件，找出基于多列组合的、仅存在于文件A中的行。

    参数:
    file_a_path (str): 第一个CSV文件的路径 (A.csv)。
    file_b_path (str): 第二个CSV文件的路径 (B.csv)。
    key_columns (list): 用于创建唯一标识符的列名列表。

    返回:
    pandas.DataFrame: 仅存在于A文件中的行。
    """
    try:
        # 优先使用 'utf-8-sig' 来处理可能由Excel生成的带有BOM的UTF-8文件
        # 如果还报错，可以尝试 'gb18030'
        encoding_type = 'utf-8-sig' 
        df_a = pd.read_csv(file_a_path, encoding=encoding_type)
        df_b = pd.read_csv(file_b_path, encoding=encoding_type)

        # 检查指定的关键列是否存在于两个文件中
        for col in key_columns:
            if col not in df_a.columns:
                return f"错误: 文件 '{file_a_path}' 中找不到列 '{col}'。"
            if col not in df_b.columns:
                return f"错误: 文件 '{file_b_path}' 中找不到列 '{col}'。"
        
        # --- 核心步骤 ---
        # 1. 创建一个临时的“复合键”，将关键列的值拼接成一个唯一的字符串
        #    使用一个不容易在正常数据中出现的的分隔符，确保唯一性
        separator = '_|||_'
        df_a['composite_key'] = df_a[key_columns].astype(str).agg(separator.join, axis=1)
        df_b['composite_key'] = df_b[key_columns].astype(str).agg(separator.join, axis=1)

        # 2. 将复合键转换为集合（Set），以便进行高效的差集运算
        set_a_keys = set(df_a['composite_key'])
        set_b_keys = set(df_b['composite_key'])

        # 3. 找出仅存在于A中的复合键
        unique_keys_in_a = set_a_keys - set_b_keys

        # 4. 根据这些唯一的复合键，从原始的df_a中筛选出对应的行
        result_df = df_a[df_a['composite_key'].isin(unique_keys_in_a)].copy()

        # 5. 删除临时的复合键列，使输出结果更干净
        result_df.drop(columns=['composite_key'], inplace=True)
        
        # 将 composite_key 列从 df_a 中也删除，以免影响原始 DataFrame (如果后续需要使用)
        df_a.drop(columns=['composite_key'], inplace=True)

        return result_df

    except FileNotFoundError as e:
        return f"错误：找不到文件 {e.filename}。"
    except Exception as e:
        # 捕获其他可能的错误，例如编码错误
        return f"处理过程中发生错误: {e}"

# --- 主程序 ---
if __name__ == "__main__":
    # --- 请在这里配置您的文件路径和列名 ---
    # 记得处理Windows路径问题，推荐使用 r"..." 原始字符串
    file_b = "C:\\Users\\20268\Desktop\\项目\数据集\\stratege_hybrid\\accuracy_hybrid.csv"
    file_a = "C:\\Users\\20268\Desktop\\项目\数据集\\stratege_point\\label_file_with_full_analysis.csv"
    columns_to_compare = ['文件名', '特征']
    output_file = 'A中独有的行.csv' # 结果将保存到这个文件

    # 执行比较
    result = find_unique_rows_in_a(file_a, file_b, columns_to_compare)

    # 打印和保存结果
    if isinstance(result, pd.DataFrame):
        if not result.empty:
            print(f"比较完成！在 '{file_a}' 中找到了 {len(result)} 行独有的数据（基于 {', '.join(columns_to_compare)} 列）。")
            print("独有的数据如下：")
            print(result)
            
            # 将结果保存到新的CSV文件
            try:
                result.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"\n结果已成功保存到文件: {output_file}")
            except Exception as e:
                print(f"\n保存文件时出错: {e}")
        else:
            print(f"在 '{file_a}' 中没有找到独有的行（基于 {', '.join(columns_to_compare)} 列）。")
    else:
        # 如果函数返回的是错误信息，则打印该信息
        print(result)