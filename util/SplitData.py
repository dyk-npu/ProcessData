import os
import shutil

# 设置你的数据文件夹路径
data_dir = 'D:\\CAD数据集\\j1.0.0\\joint'

# 文件类型与目标文件夹的映射
ext_to_folder = {
    '.json': 'json',
    '.obj': 'obj',
    '.smt': 'smt',
    '.step': 'step',
    '.png': 'png'  # 新增png类型
}

# 遍历文件夹中的所有文件
for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)
    if os.path.isfile(file_path):
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext in ext_to_folder:
            target_folder = os.path.join(data_dir, ext_to_folder[ext])
            os.makedirs(target_folder, exist_ok=True)
            shutil.move(file_path, os.path.join(target_folder, filename))

print("文件分类完成。")