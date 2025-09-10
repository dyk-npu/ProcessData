import pickle
with open(r"D:\CAD数据集\j1.0.0\joint\j1.0.0_preprocessed\j1.0.0_preprocessed\val.pickle", "rb") as f:
    data = pickle.load(f)
print(data.keys())

print("data[files]: ",data["files"][1] ) 

print("data[original_file_count]: ",(data["original_file_count"] )  )  # data["graphs"] 是一个list


print("----------------------------------------------------------")

print("data[graphs]: ",type(data["graphs"] ) )  # data["graphs"] 是一个list
print("data[graphs]: ", len(data["graphs"] ) ) # 1926

print("----------------------------------------------------------")

print("data[graphs[1][0]]: ", data["graphs"][218][0] )  # 第一个零件
print("data[graphs[1][0]]: ", data["graphs"][218][0].edge_index )  # 第一个零件 
print("----------------------------------------------------------")
# print("data[graphs[0][0]].x: ", data["graphs"][20][0].x[14] )  # 第二个零件
print("data[graphs[1][1]]: ", data["graphs"][218][1] )  # 第一个零件
print("----------------------------------------------------------")
print("data[graphs[1][1]].length: ", data["graphs"][28][1].length )  # 第二个零件
print("----------------------------------------------------------")
print("data[graphs][1][2]: ", data["graphs"][218][2] )  # 组合的Joint

print("----------------------------------------------------------")

print("data[graph_files][1][0],data[graph_files][1][1]: ", data["graph_files"][218][0] ,data["graph_files"][218][1]) # 这是一个二维的list第一个维度是图的配对的索引，第二个维度是图配对的两个文件名 str类型


