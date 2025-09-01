import pandas as pd

# --- 配置 ---
# 您的原始CSV文件名 (使用原始字符串 r"..." 来避免路径问题)
input_filename = r"C:\Users\20268\Desktop\项目\数据集\stratege_point\accuracy_point.csv"
# 处理后生成的新文件名
output_filename = r"C:\Users\20268\Desktop\项目\数据集\stratege_point\label_file_with_full_analysis.csv"


# --- 辅助函数 (保持不变) ---
def assign_accuracy_group(accuracy):
    """根据输入的准确率(百分比)，返回其所属的区间标签"""
    if 0 <= accuracy < 25:
        return '[0,25)'
    elif 25 <= accuracy < 50:
        return '[25,50)'
    elif 50 <= accuracy < 75:
        return '[50,75)'
    elif 75 <= accuracy < 90:
        return '[75,90)'
    elif 90 <= accuracy <= 100:
        return '[90,100]'
    elif accuracy > 100:
        return '>100'
    else:
        return None # 处理其他意外情况


try:
    # 假设您的CSV文件可能不是标准的UTF-8编码，尝试使用 'utf-8-sig'
    df = pd.read_csv(input_filename, encoding='utf-8-sig')

    # 检查所需的列是否存在
    required_columns = ['特征', '特征面数', '打标签数量']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"错误：输入文件必须包含以下列: {required_columns}")

    # --- 任务1: 为每一行计算百分比形式的准确率 (对所有数据进行) ---
    df['准确率(%)'] = df.apply(
        lambda row: (row['打标签数量'] / row['特征面数']) * 100 if row['特征面数'] != 0 else 0,
        axis=1
    )

    # ==================== 新增代码：筛选有效数据 ====================
    # 基于刚刚计算的准确率，创建一个新的DataFrame用于后续的指标计算
    # 这个新的DataFrame剔除了准确率 > 100% 或 == 0% 的数据
    df_filtered = df[(df['准确率(%)'] > 0) & (df['准确率(%)'] <= 100)].copy()
    
    print(f"原始数据共 {len(df)} 行。")
    print(f"筛选出 {len(df_filtered)} 行有效数据 (0% < 准确率 <= 100%) 用于计算各项指标。")
    print("-" * 50)
    # ==================== 新增代码结束 ====================


    # --- 任务2: 计算每个类别的平均准确率 (使用筛选后的数据) ---
    print("--- 每个类别的平均准确率 (基于有效数据) ---")
    # 使用 df_filtered 而不是 df
    category_mean_accuracy = df_filtered.groupby('特征')['准确率(%)'].mean()
    print(category_mean_accuracy)
    print("-" * 50)

    # --- 任务3: 统计每个类别在不同准确率区间的分布数量 (使用筛选后的数据) ---

    # 1. 定义我们期望的区间标签和顺序。由于已经排除了 >100 的数据，所以从标签中移除它
    labels = ['[0,25)', '[25,50)', '[50,75)', '[75,90)', '[90,100]']
    
    # 2. 在原始df和筛选后的df_filtered上都创建分段列
    #    在 df 上创建是为了最终保存的文件有这一列
    df['准确率分段'] = df['准确率(%)'].apply(assign_accuracy_group)
    #    在 df_filtered 上创建是为了进行后续的统计
    df_filtered['准确率分段'] = df_filtered['准确率(%)'].apply(assign_accuracy_group)

    # 3. 将 '准确率分段' 列转换为 Categorical 类型，并指定顺序
    #    这样做可以确保即使某个区间没有数据，最终的表格列顺序也是正确的。
    df_filtered['准确率分段'] = pd.Categorical(df_filtered['准确率分段'], categories=labels, ordered=True)

    # 4. 计算基础分布表 (使用筛选后的数据)
    #    使用 df_filtered 而不是 df
    accuracy_distribution = df_filtered.groupby(['特征', '准确率分段']).size().unstack(fill_value=0)
    
    # --- 任务4: 计算总和与占比，并打印最终表格 ---
    
    final_table = accuracy_distribution.copy()

    # 1. 计算“总和”行
    final_table.loc['总和'] = final_table.sum(axis=0)

    # 2. 计算“占比”行
    grand_total = final_table.loc['总和'].sum()
    if grand_total > 0:
        percentage_row = (final_table.loc['总和'] / grand_total * 100).map('{:.2f}%'.format)
    else:
        percentage_row = (final_table.loc['总和'] * 0).map('{:.2f}%'.format)

    final_table.loc['占比'] = percentage_row
    
    # 3. 打印最终的完整表格
    print("--- 每个类别在不同准确率区间的分布数量 (基于有效数据) ---")
    print(final_table)
    print("-" * 50)

    # --- 保存结果 (保存的是包含所有原始数据和分析列的完整df) ---
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n处理完成！包含完整分析的新文件已保存到: {output_filename}")


except FileNotFoundError:
    print(f"错误：找不到文件 '{input_filename}'。请检查文件名和路径是否正确。")
except Exception as e:
    print(f"处理过程中发生错误: {e}")