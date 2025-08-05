import os
import shutil


def extract_matching_files(step_source_folder, label_source_folder, data_folder, max_files=3000):
    # Create new subfolders in the 'data' folder for step and label files
    new_step_folder = os.path.join(data_folder, "step")
    new_label_folder = os.path.join(data_folder, "label")

    # Ensure these folders exist
    os.makedirs(new_step_folder, exist_ok=True)
    os.makedirs(new_label_folder, exist_ok=True)

    # Get lists of .step and .json files (without extensions)
    step_files = {os.path.splitext(f)[0] for f in os.listdir(step_source_folder) if f.endswith(".step")}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(label_source_folder) if f.endswith(".json")}

    # Find matching file names (intersection of both sets)
    matching_files = step_files.intersection(label_files)

    # Limit to max_files
    matching_files = list(matching_files)[:max_files]

    # Copy matching files to the new folders
    for filename in matching_files:
        # Copy .step file
        step_src_path = os.path.join(step_source_folder, f"{filename}.step")
        step_dst_path = os.path.join(new_step_folder, f"{filename}.step")
        shutil.copy(step_src_path, step_dst_path)
        print(f"Copied {filename}.step to '{new_step_folder}'.")

        # Copy .json file
        label_src_path = os.path.join(label_source_folder, f"{filename}.json")
        label_dst_path = os.path.join(new_label_folder, f"{filename}.json")
        shutil.copy(label_src_path, label_dst_path)
        print(f"Copied {filename}.json to '{new_label_folder}'.")


if __name__ == "__main__":
    # Define the path where the 'step' and 'label' folders are located
    base_folder = "C:\\Users\\20268\Desktop\hjc\Data2"  # 修改此处为您的基础文件夹路径，包含'step'和'label'子文件夹

    # Define the path for the new 'data' folder
    data_folder = os.path.join(base_folder, "data")

    # Source folders
    step_source_folder = os.path.join(base_folder, "step")
    label_source_folder = os.path.join(base_folder, "label")

    # Validate if the provided paths are valid directories
    if not (os.path.isdir(step_source_folder) and os.path.isdir(label_source_folder)):
        print("Invalid path. Please ensure 'step' and 'label' folders exist under the specified base folder.")
        exit(1)

    # Extract up to 3000 matching files from 'step' and 'label' folders to 'data/step' and 'data/label'
    extract_matching_files(step_source_folder, label_source_folder, data_folder, max_files=3000)

    print("File extraction completed!")