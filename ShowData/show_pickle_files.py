import pickle

file_path = 'D:/CAD数据集/j1.0.0/joint/j1.0.0_preprocessed/joint/val.pickle'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("pickle文件对象类型：", type(data))

# 如果是列表或字典，打印长度和类型
if isinstance(data, list):
    print("这是一个列表，长度为：", len(data))
    if len(data) > 0:
        print("第一个元素类型：", type(data[0]))
        print("第一个元素属性：", dir(data[0]))
elif isinstance(data, dict):
    print("这是一个字典，键有：", list(data.keys()))
    for k, v in list(data.items())[:3]:  # 只预览前3项
        print(f"键: {k} -> 类型: {type(v)}")
else:
    print("对象的属性有：", dir(data))
    # 如果对象有 __dict__ 属性，说明可以进一步查看成员
    if hasattr(data, '__dict__'):
        print("对象 __dict__ 内容预览：")
        for key in list(data.__dict__.keys())[:10]:
            print(f"  {key}: {type(data.__dict__[key])}")
    # 如果对象有 keys 方法（比如 Data 对象）
    if hasattr(data, 'keys'):
        print("对象 keys() 内容：", list(data.keys()))
