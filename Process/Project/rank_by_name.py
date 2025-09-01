import pandas as pd
import re

# 读取 CSV
df = pd.read_csv("C:\\Users\\20268\\Desktop\\项目\\数据集\\stratege_hybrid\\accuracy_hybrid.csv",encoding='gbk')

# 提取文件名中的数字部分作为排序键
df["排序键"] = df["文件名"].apply(lambda x: int(re.search(r"\d+", x).group()))

# 按排序键排序
df = df.sort_values(by="排序键")

# 去掉辅助列
df = df.drop(columns=["排序键"])

# 保存结果
df.to_csv("output.csv", index=False)

print("排序完成，结果已保存到 output.csv")
