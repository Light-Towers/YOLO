import cv2
import numpy as np
import json
import os
import shutil

# ==================== 功能1: 多边形(Polygon)转旋转框(Rotation) ====================
def convert_polygon_to_rotation(input_folder, output_folder):
    """
    将指定文件夹中所有JSON文件里的多边形(polygon)标注转换为旋转框(rotation)标注。
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"开始转换多边形 -> 旋转框: {input_folder} -> {output_folder}")

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

        print(f"  已处理: {filename}")

    print(f"✓ 多边形转换完成！结果保存在: {output_folder}\n")

# ==================== 功能2: 旋转框角度清零 (设为0度) ====================
def zero_rotation_angles(input_folder, output_folder):
    """
    将指定文件夹中所有JSON文件里的旋转框(rotation)标注的角度(direction)设置为0。
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"开始清零旋转框角度: {input_folder} -> {output_folder}")

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

        print(f"  已处理: {filename} (修改了 {modified_count} 个旋转框)")

    print(f"✓ 角度清零完成！结果保存在: {output_folder}\n")

# ==================== 功能3: 转换为水平矩形 ====================
def convert_to_horizontal_rect(input_folder, output_folder):
    """
    将旋转框转换为真正的水平矩形（修改顶点和角度）。
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"开始转换为水平矩形: {input_folder} -> {output_folder}")

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

        print(f"  已处理: {filename}")

    print(f"✓ 水平矩形转换完成！结果保存在: {output_folder}\n")

# ==================== 功能4: SAHI检测结果转LabelMe格式 ====================
def sahi_to_labelme_format(input_json_path, output_json_path, image_path=None, image_size=None, is_trun_zero=False):
    """
    将SAHI检测结果JSON文件转换为LabelMe格式的JSON文件
    参数:
    - input_json_path: 输入的SAHI JSON文件路径
    - output_json_path: 输出的LabelMe JSON文件路径
    - image_path: 对应图像文件路径（可选）
    - image_size: 图像尺寸 (width, height)，如 (1920, 1080)（可选）
    - is_trun_zero: 是否将角度清零（True时旋转坐标到0度，False时保留原始角度）
    """
    print(f"开始转换SAHI检测结果 -> LabelMe格式: {input_json_path} -> {output_json_path}")
    print(f"  参数 - is_trun_zero: {is_trun_zero}")
    
    # 读取SAHI检测结果
    with open(input_json_path, 'r', encoding='utf-8') as f:
        sahi_data = json.load(f)
    
    # 创建LabelMe格式的基本结构
    labelme_structure = {
        "version": "5.10.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_path,
        "imageData": None
    }

    # 如果提供了图像尺寸，添加到结构中
    if image_size:
        labelme_structure["imageHeight"] = image_size[1]
        labelme_structure["imageWidth"] = image_size[0]
    else:
        # 尝试从检测结果中推断图像尺寸
        max_x = max_y = 0
        for item in sahi_data:
            if "poly" in item:
                poly_points = item["poly"]
                for i in range(0, len(poly_points), 2):
                    if i + 1 < len(poly_points):
                        x, y = float(poly_points[i]), float(poly_points[i+1])
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
        if max_x > 0 and max_y > 0:
            labelme_structure["imageHeight"] = int(max_y * 1.1)  # 添加一些边缘
            labelme_structure["imageWidth"] = int(max_x * 1.1)

    # 遍历每个检测结果
    for item in sahi_data:
        if "poly" in item:  # 确保存在poly字段
            # 将poly转换为points格式
            poly_points = item["poly"]
            
            # 将一维poly数组转换为二维points数组 [[x1,y1], [x2,y2], ...]
            points = []
            for i in range(0, len(poly_points), 2):
                if i + 1 < len(poly_points):
                    points.append([float(poly_points[i]), float(poly_points[i+1])])
            
            # 将points转换为numpy数组用于计算
            pts = np.array(points, dtype=np.float32)
            
            # 计算最小外接矩形
            rect = cv2.minAreaRect(pts)
            angle_deg = rect[2]
            angle_rad = np.deg2rad(angle_deg)
            
            # 根据is_trun_zero参数决定是否旋转坐标到0度
            if is_trun_zero:
                # 计算旋转中心（图像中心）
                center = (labelme_structure["imageWidth"] / 2, labelme_structure["imageHeight"] / 2)
                # 创建旋转矩阵，旋转-rect[2]度使框水平放置
                rotation_matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
                # 旋转所有坐标点
                box = cv2.transform(pts.reshape(1, -1, 2), rotation_matrix)
                box = box.reshape(-1, 2).tolist()
                direction_value = 0.0
            else:
                # 不清零角度：直接使用最小外接矩形的4个顶点
                box = cv2.boxPoints(rect)
                box = box.reshape(-1, 2).tolist()
                direction_value = angle_rad
            
            # 创建shape对象
            shape = {
                "label": "item",  # 使用检测类别作为标签，默认为"item"
                "points": box,
                "shape_type": "rotation",  # 修改为rotation类型
                "direction": direction_value,
                "group_id": None,
                "description": "",
                "difficult": False,
                "flags": {},
                "attributes": {}
            }
            
            # 如果原始数据中有类别信息，使用它作为标签
            if "name" in item:
                # shape["label"] = item["name"]
                shape["label"] = "booth"  # 统一标签为booth
            elif "class" in item:
                # 如果class是数字，可以转换为有意义的名称或直接使用
                shape["label"] = str(item["class"])
                
            labelme_structure["shapes"].append(shape)
    
    # 保存转换后的结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_structure, f, ensure_ascii=False, indent=2)
    
    print(f"✓ SAHI转LabelMe完成！结果保存在: {output_json_path}")

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
    # convert_to_horizontal_rect(rotation_folder, horizontal_folder)
    
    # 或者，如果你只想清零角度但保持顶点不变：
    # zero_rotation_angles(rotation_folder, zero_angle_folder)
    
    # 步骤4: 将SAHI检测结果转换为LabelMe格式
    
    # # 示例1: 保留原始角度（不清零）
    # sahi_to_labelme_format(
    #     "D:/0-mingyang/code/mingyang_image_recognition/output/output_results/booth_obb_v13/booth/HongMu_sahi.json",
    #     "D:/0-mingyang/code/mingyang_image_recognition/output/output_results/booth_obb_v13/booth/HongMu_labelme.json",
    #     is_trun_zero=False
    # )
    
    # 示例2: 将角度清零（旋转坐标到0度）
    sahi_to_labelme_format(
        "D:/0-mingyang/code/mingyang_image_recognition/output/output_results/booth_obb_v13/booth/HongMu_sahi.json",
        "D:/0-mingyang/code/mingyang_image_recognition/output/test/HongMu.json",
        # is_trun_zero=True
    )
    
    # 示例3: 指定图像尺寸
    # sahi_to_labelme_format(
    #     "D:/0-mingyang/code/mingyang_image_recognition/output/output_results/booth_obb_v13/booth/HongMu_sahi.json",
    #     "D:/0-mingyang/code/mingyang_image_recognition/output/output_results/booth_obb_v13/booth/HongMu_labelme.json",
    #     image_path="D:/0-mingyang/code/mingyang_image_recognition/original_images/HongMu.png",
    #     image_size=(640, 800),
    #     is_trun_zero=True
    # )
    