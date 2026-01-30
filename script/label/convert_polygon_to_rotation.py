import cv2
import numpy as np
import json
import os
import shutil
from log_config import get_project_logger

# 获取项目logger
logger = get_project_logger('label.convert_polygon_to_rotation')

# ==================== 功能1: 多边形(Polygon)转旋转框(Rotation) ====================
def convert_polygon_to_rotation(input_folder, output_folder):
    """
    将指定文件夹中所有JSON文件里的多边形(polygon)标注转换为旋转框(rotation)标注。
    """
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"开始转换多边形 -> 旋转框: {input_folder} -> {output_folder}")

    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        new_shapes = []
        for shape in data.get('shapes', []):
            if shape['shape_type'] == 'polygon':
                # 将多边形点集转换为旋转矩形
                pts = np.array(shape['points'], dtype=np.float32)
                rect = cv2.minAreaRect(pts)
                box = cv2.boxPoints(rect)
                box = box.reshape(-1, 2).tolist()
                
                # 角度转换：OpenCV角度（度）-> 弧度
                angle_deg = rect[2]
                angle_rad = np.deg2rad(angle_deg)

                new_shape = {
                    "label": shape['label'],
                    "points": box,
                    "shape_type": "rotation",
                    "direction": angle_rad,
                    "group_id": shape.get('group_id'),
                    "difficult": shape.get('difficult', False),
                    "flags": shape.get('flags', {}),
                    "attributes": shape.get('attributes', {})
                }
                new_shapes.append(new_shape)
            else:
                # 非多边形标注原样保留
                new_shapes.append(shape)

        data['shapes'] = new_shapes
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"  已处理: {filename}")

    logger.info(f"✓ 多边形转换完成！结果保存在: {output_folder}\n")

# ==================== 功能2: 旋转框角度清零 (设为0度) ====================
def zero_rotation_angles(input_folder, output_folder):
    """
    将指定文件夹中所有JSON文件里的旋转框(rotation)标注的角度(direction)设置为0。
    """
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"开始清零旋转框角度: {input_folder} -> {output_folder}")

    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        modified_count = 0
        for shape in data.get('shapes', []):
            if shape.get('shape_type') == 'rotation':
                shape['direction'] = 0.0
                modified_count += 1

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"  已处理: {filename} (修改了 {modified_count} 个旋转框)")

    logger.info(f"✓ 角度清零完成！结果保存在: {output_folder}\n")

# ==================== 功能3: 转换为水平矩形 ====================
def convert_to_horizontal_rect(input_folder, output_folder):
    """
    将旋转框转换为真正的水平矩形（修改顶点和角度）。
    """
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"开始转换为水平矩形: {input_folder} -> {output_folder}")

    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for shape in data.get('shapes', []):
            if shape.get('shape_type') == 'rotation':
                pts = np.array(shape['points'], dtype=np.float32)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                
                # 重新排列为左上、右上、右下、左下（顺时针）
                shape['points'] = [
                    [float(x_min), float(y_min)],
                    [float(x_max), float(y_min)],
                    [float(x_max), float(y_max)],
                    [float(x_min), float(y_max)]
                ]
                shape['direction'] = 0.0

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"  已处理: {filename}")

    logger.info(f"✓ 水平矩形转换完成！结果保存在: {output_folder}\n")

# ==================== 主执行流程 ====================
if __name__ == "__main__":
    # 请根据你的实际情况修改这些路径
    # 原始数据路径（包含多边形标注的JSON文件）
    original_folder = "C:/Users/yangjinhua/Downloads/datasets/booth_images/"
    
    # 中间结果和最终结果的输出路径
    rotation_folder = "C:/Users/yangjinhua/Downloads/datasets/booth_rotated/"
    zero_angle_folder = "C:/Users/yangjinhua/Downloads/datasets/booth_zero_angle/"
    horizontal_folder = "C:/Users/yangjinhua/Downloads/datasets/booth_horizontal/"

    # ===== 根据需要选择执行以下步骤 =====
    
    # 步骤1: 将多边形转换为旋转框（如果你有polygon数据）
    # convert_polygon_to_rotation(original_folder, rotation_folder)
    
    # 步骤2: 仅将旋转框的角度设为0（不改变顶点）
    # zero_rotation_angles(rotation_folder, zero_angle_folder)
    
    # 步骤3: 转换为真正的水平矩形（会改变顶点）
    convert_to_horizontal_rect(rotation_folder, horizontal_folder)
    
    # 或者，如果你只想清零角度但保持顶点不变：
    # zero_rotation_angles(rotation_folder, zero_angle_folder)
    
    logger.info("请根据需求取消注释上面的相应函数调用。")