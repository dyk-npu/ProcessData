import os
import shutil
import pandas as pd

# --- 请根据您的实际情况修改以下路径 ---

# 1. CSV文件的完整路径
csv_file_path = "C:\\Users\\20268\\Desktop\\项目\\数据集\\label_file_acc.csv"

# 2. 存放 .step 文件的源文件夹路径
step_source_dir = "D:\\CAD数据集\\项目\\GFR_Dataset"

# 3. 存放 .pkl 标签文件的源文件夹路径
label_source_dir = "D:\CAD数据集\项目\GFR_TrainingData_Modify"

# 4. 您希望将文件复制到的主目标文件夹路径
#    脚本会自动在此文件夹下创建 'step' 和 'label' 子文件夹
main_destination_dir = "D:\CAD数据集\项目\GFR_Label"

# --- 脚本主程序 ---

def filter_and_copy_files_separated(csv_path, step_src, label_src, main_dest):
    """
    根据CSV中的准确率筛选文件，并将 .step 和 .pkl 文件分别复制到不同的子文件夹中。
    """
    # 定义并创建目标子文件夹
    dest_step_dir = os.path.join(main_dest, 'step')
    dest_label_dir = os.path.join(main_dest, 'label')

    if not os.path.exists(dest_step_dir):
        os.makedirs(dest_step_dir)
        print(f"已创建目标文件夹: {dest_step_dir}")

    if not os.path.exists(dest_label_dir):
        os.makedirs(dest_label_dir)
        print(f"已创建目标文件夹: {dest_label_dir}")

    try:
        # 读取CSV文件，并指定使用 'gbk' 编码
        df = pd.read_csv(csv_path, encoding='gbk')
        
        # --- 关键修正：清理'文件名'和'特征'列中潜在的首尾空格 ---
        df['文件名'] = df['文件名'].str.strip()
        df['特征'] = df['特征'].str.strip()

    except FileNotFoundError:
        print(f"错误：CSV文件未找到，请检查路径: {csv_path}")
        return
    except UnicodeDecodeError:
        print(f"错误：使用 'gbk' 编码失败。请尝试将CSV文件另存为 'UTF-8' 编码再试。")
        return
    except KeyError as e:
        print(f"错误：CSV文件中未找到指定的列名: {e}。请检查列标题是否为'文件名'和'特征'")
        return

    # 按“文件名”列进行分组
    grouped = df.groupby('文件名')

    # 遍历每个分组
    for file_name, group in grouped:
        # 检查该组内'准确率'列的所有值是否都大于0.95
        if (group['准确率'] > 0.95).all():
            print(f"\n[符合条件]: {file_name} 的所有子特征准确率均 > 0.95，准备复制文件...")

            # --- 复制 .step 文件 ---
            # 1. 复制主文件 (e.g., GFR_00001.step)
            source_step_main_path = os.path.join(step_src, f"{file_name}.step")
            dest_step_main_path = os.path.join(dest_step_dir, f"{file_name}.step")
            if os.path.exists(source_step_main_path):
                shutil.copy2(source_step_main_path, dest_step_main_path)
                print(f"  [成功] 已复制 step 文件: {os.path.basename(dest_step_main_path)}")
            else:
                print(f"  [警告] 源文件未找到，跳过: {source_step_main_path}")

            # 2. 复制子特征文件 (e.g., GFR_00001_rib.step)
            for index, row in group.iterrows():
                feature_name = row['特征']
                sub_file_name = f"{file_name}_{feature_name}.step"
                source_step_sub_path = os.path.join(step_src, sub_file_name)
                dest_step_sub_path = os.path.join(dest_step_dir, sub_file_name)
                if os.path.exists(source_step_sub_path):
                    shutil.copy2(source_step_sub_path, dest_step_sub_path)
                    print(f"  [成功] 已复制 step 文件: {os.path.basename(dest_step_sub_path)}")
                else:
                    print(f"  [警告] 源文件未找到，跳过: {source_step_sub_path}")

            # --- 复制 .pkl 标签文件 ---
            label_file_name = f"{file_name}.pkl"
            source_label_path = os.path.join(label_src, label_file_name)
            dest_label_path = os.path.join(dest_label_dir, label_file_name)
            if os.path.exists(source_label_path):
                shutil.copy2(source_label_path, dest_label_path)
                print(f"  [成功] 已复制 label 文件: {os.path.basename(dest_label_path)}")
            else:
                print(f"  [警告] 源文件未找到，跳过: {source_label_path}")
        
        else:
            pass

    print("\n脚本执行完毕。")


# --- 运行脚本 ---
if __name__ == "__main__":
    filter_and_copy_files_separated(csv_file_path, step_source_dir, label_source_dir, main_destination_dir)