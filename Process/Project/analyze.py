import pandas as pd
import os

# --- 配置 ---
# 请根据您的实际文件位置修改以下路径
# 任务1所需路径
INPUT_CSV_FOR_ANALYSIS = r"C:\Users\20268\Desktop\项目\数据集\stratege_hybrid_point_1.0_attribute_1.0_V2.0\acc_hybrid_point_1.0_attribute_1.0_V2.0.csv"
OUTPUT_CSV_FOR_ANALYSIS = r"C:\Users\20268\Desktop\项目\数据集\stratege_hybrid_point_1.0_attribute_1.0_V2.0\analyze2.csv"

# 任务2所需路径
CSV_FOR_COUNTING = INPUT_CSV_FOR_ANALYSIS


# =================================================================================
# 函数1：分析准确率分布 (已根据“剔除0%”的意图修正)
# =================================================================================
def analyze_accuracy_distribution(input_filename, output_filename):
    """
    读取原始数据文件，计算准确率，排除0%的数据后，分析剩余数据的分布情况，
    并将带有分析结果的完整数据保存到新文件。
    """
    print("--- 开始执行任务1：分析准确率分布 (已排除0%数据) ---")
    
    # --- 辅助函数 ---
    def assign_accuracy_group(accuracy):
        """根据输入的准确率(百分比)，返回其所属的区间标签"""
        # ★★★ 修改点1：将第一个区间的条件从 0 <= 改为 0 <，以精确匹配筛选后的数据 ★★★
        if 0 < accuracy < 25:
            return '(0,25)' # 返回的标签是圆括号
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
            return None

    try:
        if not os.path.exists(input_filename):
            print(f"错误：找不到输入文件 '{input_filename}'。请检查路径是否正确。")
            return

        df = pd.read_csv(input_filename, encoding='utf-8-sig')

        required_columns = ['特征', '特征面数', '打标签数量']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"错误：输入文件必须包含以下列: {required_columns}")

        df['准确率(%)'] = df.apply(
            lambda row: (row['打标签数量'] / row['特征面数']) * 100 if row['特征面数'] != 0 else 0,
            axis=1
        )

        # 核心逻辑：保持 > 0 的筛选条件，这完全符合您的“剔除0%数据”的要求
        df_filtered = df[(df['准确率(%)'] > 0) & (df['准确率(%)'] <= 100)].copy()
        
        print(f"原始数据共 {len(df)} 行。")
        print(f"筛选出 {len(df_filtered)} 行有效数据 (0% < 准确率 <= 100%) 用于计算各项指标。")
        print("-" * 50)

        if not df_filtered.empty:
            print("--- 每个类别的平均准确率 (基于有效数据) ---")
            category_mean_accuracy = df_filtered.groupby('特征')['准确率(%)'].mean()
            print(category_mean_accuracy)
            print("-" * 50)
        else:
            print("--- 没有找到有效数据来计算每个类别的平均准确率 ---")

        # ★★★ 核心修改点2：让 labels 列表中的标签与函数返回的标签完全匹配 ★★★
        labels = ['(0,25)', '[25,50)', '[50,75)', '[75,90)', '[90,100]']
        
        df['准确率分段'] = df['准确率(%)'].apply(assign_accuracy_group)
        df_filtered['准确率分段'] = df_filtered['准确率(%)'].apply(assign_accuracy_group)
        df_filtered['准确率分段'] = pd.Categorical(df_filtered['准确率分段'], categories=labels, ordered=True)

        accuracy_distribution = df_filtered.groupby(['特征', '准确率分段'], observed=False).size().unstack(fill_value=0)
        
        final_table = accuracy_distribution.copy()
        final_table.loc['总和'] = final_table.sum(axis=0)
        grand_total = final_table.loc['总和'].sum()
        if grand_total > 0:
            percentage_row = (final_table.loc['总和'] / grand_total * 100).map('{:.2f}%'.format)
        else:
            percentage_row = (final_table.loc['总和'] * 0).map('{:.2f}%'.format)
        final_table.loc['占比'] = percentage_row
        
        print("--- 每个类别在不同准确率区间的分布数量 (基于有效数据) ---")
        print(final_table)
        print("-" * 50)

        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"任务1完成！包含完整分析的新文件已保存到: {output_filename}\n")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")


# =================================================================================
# 函数2：统计不同准确率条件的文件 (这个函数逻辑原本就是正确的，无需改动)
# =================================================================================
def analyze_file_accuracy_groups(csv_path):
    """
    读取CSV文件，按文件名分组，统计满足四种不同条件的文件数量。
    """
    print("--- 开始执行任务2：按文件统计各类准确率 ---")
    
    try:
        if not os.path.exists(csv_path):
            print(f"错误：找不到输入文件 '{csv_path}'。请检查路径是否正确。")
            return

        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        required_cols_task2 = ['文件名', '特征', '准确率']
        if not all(col in df.columns for col in required_cols_task2):
            if '准确率(%)' in df.columns:
                df.rename(columns={'准确率(%)': '准确率'}, inplace=True)
            else:
                 raise ValueError(f"错误：任务2所需文件必须包含以下列: {required_cols_task2}")

        df['文件名'] = df['文件名'].astype(str).str.strip()
        df['特征'] = df['特征'].astype(str).str.strip()
        
        if df['准确率'].dtype == 'object':
             df['准确率'] = df['准确率'].str.replace('%', '', regex=False).astype(float)
        
        # 统一单位为 0-1 之间的小数
        if df['准确率'].max() > 1:
            df['准确率'] = df['准确率'] / 100.0
        
    except Exception as e:
        print(f"读取或预处理 CSV '{csv_path}' 时出错: {e}")
        return

    grouped = df.groupby('文件名')

    high_accuracy_files_count = 0
    normal_files_count = 0
    zero_accuracy_files_count = 0
    over_100_files_count = 0
    
    for file_name, group in grouped:
        if ((group['准确率'] >= 0.95) & (group['准确率'] <= 1.0)).all():
            high_accuracy_files_count += 1
        
        if ((group['准确率'] > 0) & (group['准确率'] <= 1.0)).all():
            normal_files_count += 1

        if (group['准确率'] == 0).any():
            zero_accuracy_files_count += 1
            
        if (group['准确率'] > 1.0).any():
            over_100_files_count += 1

    print(f"高精度文件 (所有子特征准确率 >= 95%) 的总数: {high_accuracy_files_count}")
    print(f"正常基础文件 (所有子特征准确率 > 0% 且 <= 100%) 的总数: {normal_files_count}")
    print(f"含0%准确率子特征的文件总数 (至少有一个0%): {zero_accuracy_files_count}")
    print(f"含超100%准确率子特征的文件总数 (至少有一个>100%): {over_100_files_count}")
    
    print("任务2完成！\n")


# --- 主程序入口 ---
if __name__ == "__main__":
    analyze_accuracy_distribution(INPUT_CSV_FOR_ANALYSIS, OUTPUT_CSV_FOR_ANALYSIS)
    analyze_file_accuracy_groups(OUTPUT_CSV_FOR_ANALYSIS)