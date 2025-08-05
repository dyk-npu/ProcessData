import os
import shutil

# Define the source folder path directly in the script
data_path = "C:\\Users\\20268\Desktop\hjc\Data2"  # 修改此处为您的文件夹路径

def separate_files(source_folder):
    # Define target folder names
    step_folder = os.path.join(source_folder, "step")
    label_folder = os.path.join(source_folder, "label")

    # Create target folders if they don't exist
    os.makedirs(step_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)

        # Ensure only files are processed (ignore subfolders)
        if os.path.isfile(file_path):
            if filename.endswith(".step"):
                # Move .step files to the 'step' folder
                shutil.move(file_path, os.path.join(step_folder, filename))
                print(f"Moved {filename} to 'step' folder.")
            elif filename.endswith(".json"):
                # Move .json files to the 'label' folder
                shutil.move(file_path, os.path.join(label_folder, filename))
                print(f"Moved {filename} to 'label' folder.")

if __name__ == "__main__":
    # Validate if the provided data_path is a valid directory
    if os.path.isdir(data_path):
        separate_files(data_path)
        print("File separation completed!")
    else:
        print("Invalid path. Please modify the 'data_path' variable with a valid folder path.")