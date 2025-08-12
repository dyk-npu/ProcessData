
import pickle
import numpy as np


def load_labels(label_filename):
    with open(label_filename, "rb") as f:
        labels = pickle.load(f)
    return np.array(labels)


data = load_labels("D:/CAD数据集/项目/GFR_TrainingData_Modify/GFR_00013.pkl")

dict = data.item()
label = dict['face_labels']

print("label:",label)

result = [int(k) for k, v in label.items() if int(v) == 1]
print("result:", result)