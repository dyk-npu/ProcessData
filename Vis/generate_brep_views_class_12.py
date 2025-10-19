import os
import math
import pathlib
from tqdm import tqdm

# --- 重要：设置环境变量以启用无头（offscreen）渲染 ---
# 这允许脚本在没有图形界面的服务器上运行
os.environ['PYTHONOCC_OFFSCREEN_RENDERER'] = 'EGL'

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.OCCViewer import rgb_color
from OCC.Display.SimpleGui import init_display
from OCC.Core.Bnd import Bnd_Box
from OCC.Core import BRepBndLib
import shutup

shutup.please()  # 静音警告信息


# --- 视图配置 ---
NUM_VIEWS = 12
HORIZONTAL_ANGLE_STEP = 360 / NUM_VIEWS
VERTICAL_ANGLE_DEG = 90
# ---------------------


def set_custom_camera(display, horizontal_angle_deg, vertical_angle_deg, shape_center, shape_diagonal):
    """
    (此函数保持不变)
    设置一个自定义的相机视角。
    """
    h_angle_rad = math.radians(horizontal_angle_deg)
    v_angle_rad = math.radians(vertical_angle_deg)
    distance = shape_diagonal * 1.5
    center_x, center_y, center_z = shape_center
    eye_x = center_x + distance * math.cos(v_angle_rad) * math.sin(h_angle_rad)
    eye_y = center_y - distance * math.cos(v_angle_rad) * math.cos(h_angle_rad)
    eye_z = center_z + distance * math.sin(v_angle_rad)
    view = display.View
    view.SetEye(eye_x, eye_y, eye_z)
    view.SetAt(center_x, center_y, center_z)
    view.SetUp(0, 0, 1)


def generate_views_for_shape(display, shape, output_dir):
    """
    (此函数保持不变)
    为已加载的shape生成并保存12个环绕视图。
    """
    bbox = Bnd_Box()
    BRepBndLib.brepbndlib_Add(shape, bbox) 
    
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    if not bbox.IsVoid():
        center_point = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
        diagonal_length = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)
    else: 
        center_point = (0, 0, 0)
        diagonal_length = 1.0

    output_dir_str = str(output_dir)
    file_stem = output_dir.name

    for i in range(NUM_VIEWS):
        horizontal_angle = i * HORIZONTAL_ANGLE_STEP
        set_custom_camera(display, horizontal_angle, VERTICAL_ANGLE_DEG, center_point, diagonal_length)
        display.FitAll()
        display.View.Redraw()
        output_filename = os.path.join(output_dir_str, f"{file_stem}_{i:02d}.png")
        display.View.Dump(output_filename)
    
    tqdm.write(f"  -> 已为 '{file_stem}' 生成 {NUM_VIEWS} 个视图。")


def process_recursively(root_input_dir):
    """
    (函数已修改)
    递归查找所有STEP文件，并根据指定的输出格式生成图片。
    """
    input_path = pathlib.Path(root_input_dir)

    if not input_path.is_dir():
        print(f"错误：输入文件夹不存在 -> {input_path}")
        return

    # --- 新增：根据输入根目录直接构建输出根目录 ---
    # 例如将 '...\step_files' 替换为 '...\image_files_12'
    if 'step_files' not in str(input_path):
         print(f"警告：输入根目录 '{input_path}' 不包含 'step_files'。请检查路径。")
         # 您可以根据需要决定是退出还是继续
    output_root_path_str = str(input_path).replace('step_files', 'image_files_12', 1)
    output_root_path = pathlib.Path(output_root_path_str)
    print(f"所有图片将被保存在根目录: {output_root_path}")
    # --- 修改结束 ---

    print(f"正在从 '{input_path}' 及其子目录中搜索文件...")
    step_files = list(input_path.rglob("*.st*p"))

    if not step_files:
        print("未找到任何 STEP 文件。")
        return
        
    print(f"找到 {len(step_files)} 个文件待处理。")

    display, start_display, add_menu, add_function_to_menu = init_display()
    display.View.TriedronErase()
    display.set_bg_gradient_color(rgb_color(1, 1, 1), rgb_color(1, 1, 1))
    display.View.SetShadingModel(1)

    for step_file in tqdm(step_files, desc="总进度"):
        try:
            # --- 修改：构建输出路径以匹配您的要求 ---
            file_stem = step_file.stem  # 例如 'a_ABeeZee_lower'
            # 输出子目录为: ...\image_files_12\a_ABeeZee_lower
            output_subdir = output_root_path / file_stem
            # --- 修改结束 ---
            
            # 检查是否所有视图都已存在
            all_views_exist = True
            if output_subdir.exists():
                last_view_file = output_subdir / f"{file_stem}_{NUM_VIEWS-1:02d}.png"
                if not last_view_file.exists():
                    all_views_exist = False
            else:
                all_views_exist = False

            if all_views_exist:
                tqdm.write(f"所有视图均已存在，跳过: {file_stem}")
                continue

            output_subdir.mkdir(parents=True, exist_ok=True)
            
            display.EraseAll()
            
            step_reader = STEPControl_Reader()
            if step_reader.ReadFile(str(step_file)) != 1:
                tqdm.write(f"警告：无法读取文件 -> {step_file}")
                continue
                
            step_reader.TransferRoots()
            shape = step_reader.OneShape()

            if shape is None or shape.IsNull():
                tqdm.write(f"警告：在文件中未找到有效实体 -> {step_file}")
                continue

            display.DisplayShape(shape, update=True, color=rgb_color(0.4, 0.4, 0.45))
            generate_views_for_shape(display, shape, output_subdir)
        
        except Exception as e:
            tqdm.write(f"处理文件 {step_file} 时发生严重错误: {e}")

    print("\n所有文件处理完毕！")


if __name__ == '__main__':
    # --- 请在这里配置您的STEP文件所在的根目录 ---
    # 脚本会根据这个路径自动生成 'image_files_12' 输出目录
    input_root_folder = r"E:\CAD数据集\SolidLetters\step\step_files" 
    # ---------------------------------------------
    process_recursively(input_root_folder)