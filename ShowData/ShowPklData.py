
import pickle
import numpy as np


def load_labels(label_filename):
    with open(label_filename, "rb") as f:
        labels = pickle.load(f)
    return np.array(labels)


data = load_labels("D:/CAD数据集/项目/GFR_TrainingData_Modify/GFR_01747.pkl")

dict = data.item()
label = dict['face_labels']
print("The label of the GFR_00023 is:", label)


# Categorize feature indices according to their assigned semantic labels
result_base = [int(k) for k, v in label.items() if int(v) == 0]      # Label 0: Base-type features
result_clip = [int(k) for k, v in label.items() if int(v) == 1]      # Label 1: Clip-type features
result_boss = [int(k) for k, v in label.items() if int(v) == 2]      # Label 2: Boss-type features
result_rib = [int(k) for k, v in label.items() if int(v) == 3]       # Label 3: Rib-type features
result_contact = [int(k) for k, v in label.items() if int(v) == 4]   # Label 4: Contact-type features

# Output the categorized results in a structured form
print("Indices of base-type complex features (Label 0):", result_base)
print("Indices of clip-type complex features (Label 1):", result_clip)
print("Indices of boss-type complex features (Label 2):", result_boss)
print("Indices of rib-type complex features (Label 3):", result_rib)
print("Indices of contact-type complex features (Label 4):", result_contact)
