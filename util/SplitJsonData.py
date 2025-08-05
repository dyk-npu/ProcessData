import os
import shutil

# 你的 json 文件夹路径（可根据实际情况修改）
json_dir = 'D:\CAD数据集\j1.0.0\joint\json'

# 新文件夹名，与 json 文件夹同级
new_dir = os.path.join(os.path.dirname(json_dir), 'joint_set_json')

# 如果新文件夹不存在则创建
os.makedirs(new_dir, exist_ok=True)

# 遍历 json 文件夹中的所有文件
for filename in os.listdir(json_dir):
    # 检查是否以 joint_set_ 开头并且以 .json 结尾
    if filename.startswith('joint_set_') and filename.endswith('.json'):
        src = os.path.join(json_dir, filename)
        dst = os.path.join(new_dir, filename)
        # 移动文件
        shutil.move(src, dst)
        # 如果你想保留原文件，可改成 shutil.copy(src, dst)

print('所有 joint_set_*.json 文件已移动到 joint_set_json 文件夹。')
