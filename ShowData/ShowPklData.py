
import pickle
import numpy as np


def load_labels(label_filename):
    with open(label_filename, "rb") as f:
        labels = pickle.load(f)
    return np.array(labels)


data = load_labels("D:/CAD数据集/项目/GFR_TrainingData_Modify/GFR_00017.pkl")



print("data内容类型:", type(data.item()))

print("keys:", data.item()['face_labels'])
