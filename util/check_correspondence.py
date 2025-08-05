import os

def check_folders_correspondence(folder1, folder2):
    # Get the list of files in both folders (without extensions)
    files_folder1 = {os.path.splitext(f)[0] for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))}
    files_folder2 = {os.path.splitext(f)[0] for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))}

    # Check if the sets of file names are identical
    if files_folder1 == files_folder2:
        print("The two folders contain exactly corresponding files.")
        return True
    else:
        # Find missing files in each folder
        missing_in_folder1 = files_folder2 - files_folder1
        missing_in_folder2 = files_folder1 - files_folder2

        if missing_in_folder1:
            print(f"Files missing in folder 1 ({folder1}):")
            for filename in sorted(missing_in_folder1):
                print(f"  - {filename}")
        if missing_in_folder2:
            print(f"Files missing in folder 2 ({folder2}):")
            for filename in sorted(missing_in_folder2):
                print(f"  - {filename}")

        print("The two folders do not contain completely corresponding files.")
        return False

if __name__ == "__main__":
    # Define the paths to the two folders you want to compare
    folder1 = 'C:\\Users\\20268\Desktop\hjc\Data2\label'  # 修改为第一个文件夹路径
    folder2 = 'C:\\Users\\20268\Desktop\hjc\Data2\step'  # 修改为第二个文件夹路径

    # Validate if the provided paths are valid directories
    if not (os.path.isdir(folder1) and os.path.isdir(folder2)):
        print("Invalid path. Please ensure both folder paths are valid.")
        exit(1)

    # Check correspondence between the two folders
    check_folders_correspondence(folder1, folder2)