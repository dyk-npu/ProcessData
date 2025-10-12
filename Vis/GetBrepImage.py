import os
import math # 导入数学库用于三角函数计算
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.OCCViewer import rgb_color
from OCC.Display.SimpleGui import init_display
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add

def set_custom_camera(display, horizontal_angle_deg, vertical_angle_deg, shape_center, shape_diagonal):
    """
    设置一个自定义的相机视角。

    Args:
        display: OCC的显示对象。
        horizontal_angle_deg (float): 水平环绕角度 (0-360)。
        vertical_angle_deg (float): 垂直俯视角度 (0-90)。
        shape_center (tuple): 模型的中心点 (x, y, z)。
        shape_diagonal (float): 模型包围盒的对角线长度，用于计算距离。
    """
    # 1. 将角度从度转换为弧度
    h_angle_rad = math.radians(horizontal_angle_deg)
    v_angle_rad = math.radians(vertical_angle_deg)

    # 2. 计算一个合适的相机距离
    # 乘以1.5倍的对角线长度可以确保模型完整显示且有留白
    distance = shape_diagonal * 1.5

    # 3. 使用球坐标计算相机位置 (Eye Point)
    # 相对于模型中心点进行计算
    center_x, center_y, center_z = shape_center
    eye_x = center_x + distance * math.cos(v_angle_rad) * math.sin(h_angle_rad)
    eye_y = center_y - distance * math.cos(v_angle_rad) * math.cos(h_angle_rad) # Y轴反向，使0度朝前
    eye_z = center_z + distance * math.sin(v_angle_rad)

    # 4. 设置相机
    view = display.View
    view.SetEye(eye_x, eye_y, eye_z)  # 设置相机位置
    view.SetAt(center_x, center_y, center_z)    # 设置目标点
    view.SetUp(0, 0, 1)                         # 设置上方向为Z轴

def generate_cad_views(step_file_path, output_dir):
    """
    读取STEP文件并从自定义的俯视角度生成2D视图。
    """
    if not os.path.exists(step_file_path):
        print(f"错误：输入文件不存在 -> {step_file_path}")
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"错误：无法创建输出目录 -> {output_dir}\n{e}")
        return
    
    display, start_display, add_menu, add_function_to_menu = init_display()
    
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file_path)
    if status != 1: return
    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    # --- 计算模型的中心和尺寸，为设置相机做准备 ---
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    center_point = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    diagonal_length = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)

    # --- 使用推荐的银灰色 ---
    display.DisplayShape(shape, update=True, color=rgb_color(0.4, 0.4, 0.45))
    display.set_bg_gradient_color(rgb_color(1,1,1), rgb_color(1,1,1))
    display.View.SetShadingModel(1)
    display.Repaint()
    
    print(f"成功加载模型: {os.path.basename(step_file_path)}")

    # ==========================================================
    # ---      在这里定义您想要的拍照角度 (水平角度)       ---
    # ---      垂直角度将固定为您期望的 45 度俯视         ---
    # ==========================================================
    views_to_generate = {
        "front_right_iso": 45,   # 右前方45度视角
        "front_left_iso": -45,   # 左前方45度视角
        "back_iso": 180,         # 正后方视角
    }
    
    base_filename = os.path.splitext(os.path.basename(step_file_path))[0]

    for view_name, horizontal_angle in views_to_generate.items():
        print(f"  -> 正在生成 {view_name} (水平{horizontal_angle}°, 俯视45°) 视图...")
        
        # 调用新函数来设置自定义相机
        set_custom_camera(display, horizontal_angle, 45, center_point, diagonal_length)
        
        # FitAll在这里用于调整缩放，而不是改变角度
        display.FitAll()
        display.View.Redraw()

        output_filename = os.path.join(output_dir, f"{base_filename}_{view_name}.png")
        display.View.Dump(output_filename)
        print(f"     视图已保存到: {output_filename}")

    print("\n所有视图已成功生成！")


if __name__ == '__main__':
    
    # --- 请在这里修改您的输入和输出路径 ---
    input_step_file = r"E:\CAD数据集\SolidLetters\step\step_files\upper\a_ABeeZee_upper.step"
    output_directory = r"C:\Users\20268\Desktop\Project\MFR\ProcessData\Vis"
    
    generate_cad_views(input_step_file, output_directory)