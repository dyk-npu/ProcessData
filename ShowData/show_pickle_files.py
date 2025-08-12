import pickle

def explore(data, indent=0):
    """
    递归打印数据结构
    """
    prefix = "  " * indent
    if isinstance(data, dict):
        print(f"{prefix}dict, {len(data)} keys")
        for k, v in data.items():
            print(f"{prefix}  key: {k} -> type: {type(v).__name__}")
            explore(v, indent + 2)
    elif isinstance(data, list):
        print(f"{prefix}list, len={len(data)}")
        if len(data) > 0:
            print(f"{prefix}  first item type: {type(data[0]).__name__}")
            explore(data[0], indent + 2)
    else:
        print(f"{prefix}{type(data).__name__}, value={repr(data)[:100]}")

# 读取 pkl 文件
with open("D:/CAD数据集/项目/GFR_TrainingData_Modify/GFR_00013.pkl", "rb") as f:
    obj = pickle.load(f)

print(f"顶层类型: {type(obj).__name__}")
explore(obj)
