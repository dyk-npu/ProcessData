import pyvista as pv
import os
import time

def generate_three_views(model_path, output_dir, log_file_path):
    """
    从单个模型文件生成三视图，并包含完整的错误处理和日志记录。

    参数:
        model_path (str): 输入模型文件的完整路径。
        output_dir (str): 保存输出图像的文件夹路径。
        log_file_path (str): 错误日志文件的路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(model_path))[0]

    front_view_path = os.path.join(output_dir, f"{base_name}_front.png")
    top_view_path = os.path.join(output_dir, f"{base_name}_top.png")
    left_view_path = os.path.join(output_dir, f"{base_name}_left.png")

    if os.path.exists(front_view_path) and os.path.exists(top_view_path) and os.path.exists(left_view_path):
        print(f"  -> 已跳过 (文件已存在): {base_name}")
        return

    # --- 核心容错功能 ---
    try:
        # 尝试读取并渲染模型
        mesh = pv.read(model_path)
        
        # 检查mesh是否有效，有些损坏文件可能读出来是空的
        if mesh.n_points == 0:
            # 手动引发一个错误，这样可以被下面的except捕获并记录
            raise ValueError("Mesh is empty or corrupt, contains no points.")

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh, color='lightblue', smooth_shading=True)
        plotter.enable_parallel_projection()

        plotter.view_xz() # 前视图
        plotter.reset_camera()
        plotter.screenshot(front_view_path)

        plotter.view_xy() # 顶视图
        plotter.reset_camera()
        plotter.screenshot(top_view_path)

        plotter.view_yz() # 左视图
        plotter.reset_camera()
        plotter.screenshot(left_view_path)
        
        plotter.close()
        print(f"  -> 已成功生成平滑视图: {base_name}")

    except Exception as e:
        # 如果try块中的任何地方发生错误，则执行这里
        print(f"  -> 错误: 处理文件 '{base_name}' 失败。正在记录错误...")
        # 以追加模式('a')打开日志文件，这样不会覆盖旧的错误记录
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"文件路径: {model_path}\n")
            f.write(f"错误信息: {e}\n")
            f.write("-" * 50 + "\n")
        # 错误被处理后，程序将继续运行
        return


def process_dataset(input_dir, output_dir):
    """
    遍历整个数据集目录，为每个3D模型文件生成三视图。
    """
    print(f"开始处理数据集...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 定义日志文件路径，存放在image目录下
    # os.path.dirname(output_dir) 会获取 '.../image/train' 的上一级目录 '.../image'
    log_file = os.path.join(os.path.dirname(output_dir), 'error_log.txt')
    print(f"错误日志将记录在: {log_file}")
    print("-" * 30)
    
    start_time = time.time()
    file_count = 0

    for dirpath, _, filenames in os.walk(input_dir):
        model_files = [f for f in filenames if f.lower().endswith(('.stl', '.obj'))]
        
        if not model_files:
            continue

        relative_path = os.path.relpath(dirpath, input_dir)
        current_output_dir = os.path.join(output_dir, relative_path)
        
        print(f"\n正在处理文件夹: {relative_path}")

        for filename in model_files:
            file_count += 1
            model_file_path = os.path.join(dirpath, filename)
            # 将日志文件路径传递给处理函数
            generate_three_views(model_file_path, current_output_dir, log_file)

    end_time = time.time()
    print("-" * 30)
    print(f"处理完成！")
    print(f"总共检查了 {file_count} 个模型文件。")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")


if __name__ == '__main__':
    # --- 请在这里配置你的路径 ---
    INPUT_BASE_DIR = r"E:\CAD数据集\MCB\MCB_A"

    # --- 脚本会自动配置以下路径 ---
    train_input_dir = os.path.join(INPUT_BASE_DIR, 'train')
    train_output_dir = os.path.join(INPUT_BASE_DIR, 'image', 'train')

    if not os.path.isdir(train_input_dir):
        print(f"错误：输入的训练目录不存在，请检查路径！")
        print(f"检查路径: {train_input_dir}")
    else:
        process_dataset(train_input_dir, train_output_dir)