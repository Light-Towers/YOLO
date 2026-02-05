import cv2
import numpy as np
import json
import os
import shutil
import traceback  # 添加traceback模块用于输出完整异常信息


def _read_json_file(file_path):
    """读取JSON文件，添加异常处理"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：文件不存在 - {file_path}")
        print(traceback.format_exc())
        raise
    except json.JSONDecodeError:
        print(f"错误：JSON格式不正确 - {file_path}")
        print(traceback.format_exc())
        raise
    except Exception as e:
        print(f"错误：读取文件失败 - {file_path}, 错误: {str(e)}")
        print(traceback.format_exc())
        raise


def _write_json_file(file_path, data):
    """写入JSON文件，添加异常处理"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"错误：写入文件失败 - {file_path}, 错误: {str(e)}")
        print(traceback.format_exc())
        raise


def _get_image_size(data, json_file_path=None):
    """从JSON数据中推断图像尺寸，如果无法从数据中推断，则尝试从对应图像文件获取"""
    max_x = max_y = 0
    for shape in data.get('shapes', []):
        points = shape.get('points', [])
        for point in points:
            x, y = point
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    if max_x > 0 and max_y > 0:
        return int(max_x * 1.1), int(max_y * 1.1)
    
    # 如果JSON中有直接的图像尺寸信息，则返回这些信息
    if 'imageHeight' in data and 'imageWidth' in data:
        return int(data['imageWidth']), int(data['imageHeight'])
    
    # 如果无法从JSON数据中获得尺寸，尝试从对应的图像文件获取
    if json_file_path and os.path.exists(json_file_path):
        # 生成可能的图像文件路径（去除.json扩展名，替换为常见图像格式）
        base_path = json_file_path.rsplit('.', 1)[0]  # 移除.json扩展名
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        for ext in image_extensions:
            image_path = base_path + ext
            if os.path.exists(image_path):
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        height, width = image.shape[:2]
                        return width, height
                except Exception as e:
                    print(f"警告：无法读取图像文件 {image_path}: {str(e)}")
                    continue
                
                # 如果cv2.imread失败，尝试使用其他方法打开图像
                try:
                    import PIL.Image
                    pil_img = PIL.Image.open(image_path)
                    return pil_img.size  # 返回 (width, height)
                except ImportError:
                    pass  # 如果没有安装PIL/Pillow则跳过
                except Exception as e:
                    print(f"警告：无法使用PIL读取图像文件 {image_path}: {str(e)}")
                    continue
    
    return None


def _create_shape_from_polygon(shape):
    """将多边形转换为旋转框形状"""
    pts = np.array(shape['points'], dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = box.reshape(-1, 2).tolist()
    
    angle_deg = rect[2]
    angle_rad = np.deg2rad(angle_deg)
    
    return {
        "label": shape['label'],
        "points": box,
        "shape_type": "rotation",
        "direction": angle_rad,
        "group_id": shape.get('group_id'),
        "difficult": shape.get('difficult', False),
        "flags": shape.get('flags', {}),
        "attributes": shape.get('attributes', {})
    }


def _convert_polygon_to_rotation(input_folder, output_folder):
    """将多边形标注转换为旋转框标注"""
    os.makedirs(output_folder, exist_ok=True)
    print(f"开始转换多边形 -> 旋转框: {input_folder} -> {output_folder}")

    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            data = _read_json_file(input_path)
            new_shapes = []

            for shape in data.get('shapes', []):
                if shape['shape_type'] == 'polygon':
                    new_shapes.append(_create_shape_from_polygon(shape))
                else:
                    new_shapes.append(shape)

            data['shapes'] = new_shapes
            _write_json_file(output_path, data)
            print(f"  已处理: {filename}")

        except Exception:
            print(f"  跳过: {filename} (处理失败)")
            print(traceback.format_exc())  # 输出完整异常堆栈
            continue

    print(f"✓ 多边形转换完成！结果保存在: {output_folder}\n")


def _zero_rotation_angles(input_folder, output_folder):
    """将旋转框角度设为0"""
    os.makedirs(output_folder, exist_ok=True)
    print(f"开始清零旋转框角度: {input_folder} -> {output_folder}")

    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            data = _read_json_file(input_path)
            modified_count = 0

            for shape in data.get('shapes', []):
                if shape.get('shape_type') == 'rotation':
                    shape['direction'] = 0.0
                    modified_count += 1

            _write_json_file(output_path, data)
            print(f"  已处理: {filename} (修改了 {modified_count} 个旋转框)")

        except Exception:
            print(f"  跳过: {filename} (处理失败)")
            print(traceback.format_exc())  # 输出完整异常堆栈
            continue

    print(f"✓ 角度清零完成！结果保存在: {output_folder}\n")


def _convert_to_horizontal_rect(input_folder, output_folder):
    """将旋转框转换为水平矩形"""
    os.makedirs(output_folder, exist_ok=True)
    print(f"开始转换为水平矩形: {input_folder} -> {output_folder}")

    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            data = _read_json_file(input_path)

            for shape in data.get('shapes', []):
                if shape.get('shape_type') == 'rotation':
                    pts = np.array(shape['points'], dtype=np.float32)
                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)
                    
                    shape['points'] = [
                        [float(x_min), float(y_min)],
                        [float(x_max), float(y_min)],
                        [float(x_max), float(y_max)],
                        [float(x_min), float(y_max)]
                    ]
                    shape['direction'] = 0.0

            _write_json_file(output_path, data)
            print(f"  已处理: {filename}")

        except Exception:
            print(f"  跳过: {filename} (处理失败)")
            print(traceback.format_exc())  # 输出完整异常堆栈
            continue

    print(f"✓ 水平矩形转换完成！结果保存在: {output_folder}\n")


def sahi_to_labelme_format(input_json_path, output_json_path, is_trun_zero=False):
    """
    将SAHI检测结果JSON文件转换为LabelMe格式的JSON文件
    
    参数:
    - input_json_path: 输入的SAHI JSON文件路径
    - output_json_path: 输出的LabelMe JSON文件路径
    - is_trun_zero: 是否将角度清零（True时旋转坐标到0度，False时保留原始角度）
    """
    try:
        print(f"开始转换SAHI检测结果 -> LabelMe格式: {input_json_path} -> {output_json_path}")
        print(f"  参数 - is_trun_zero: {is_trun_zero}")
        
        # 读取SAHI检测结果
        sahi_data = _read_json_file(input_json_path)
        
        # 创建LabelMe格式的基本结构
        labelme_structure = {
            "version": "3.3.6",
            "flags": {},
            "shapes": [],
            # "imagePath": os.path.basename(input_json_path).replace('.json', ''),
            "imagePath": "HongMu.png",
            "imageData": None
        }

        # 自动获取图像尺寸
        image_size = _get_image_size(labelme_structure, input_json_path)
        if image_size:
            labelme_structure["imageHeight"] = image_size[1]
            labelme_structure["imageWidth"] = image_size[0]

        # 遍历每个检测结果
        for item in sahi_data:
            if "poly" not in item:
                continue

            # 将poly转换为points格式
            poly_points = item["poly"]
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
                # # 计算旋转中心（图像中心）
                # center = (labelme_structure["imageWidth"] / 2, labelme_structure["imageHeight"] / 2)
                # # 创建旋转矩阵，旋转-rect[2]度使框水平放置
                # rotation_matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
                # # 旋转所有坐标点
                # box = cv2.transform(pts.reshape(1, -1, 2), rotation_matrix)
                # box = box.reshape(-1, 2).tolist()
                # direction_value = 0.0


                # 【修复逻辑】：不旋转坐标系，而是直接提取该多边形的水平最大边界
                # 这样生成的框永远是正的，且坐标永远在图片内部
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)
                
                # 构建水平矩形的四个点（左上，右上，右下，左下）
                box = [
                    [float(x_min), float(y_min)],
                    [float(x_max), float(y_min)],
                    [float(x_max), float(y_max)],
                    [float(x_min), float(y_max)]
                ]
                direction_value = 0.0
            else:
                # 保持原始旋转状态
                rect = cv2.minAreaRect(pts)
                angle_deg = rect[2]
                box = cv2.boxPoints(rect).tolist()
                direction_value = np.deg2rad(angle_deg)

            
            # 最终安全检查：防止由于浮点运算导致的微小越界
            if image_size:
                w, h = image_size
                box = [[max(0, min(p[0], w)), max(0, min(p[1], h))] for p in box]

            
            # 创建shape对象，label写死为"booth"
            shape = {
                "kie_linking": [],
                "label": "booth",
                "points": box,
                "shape_type": "rotation",
                "direction": direction_value,
                "group_id": None,
                "description": "",
                "difficult": False,
                "flags": {},
                "attributes": {}
            }
            
            labelme_structure["shapes"].append(shape)
        
        # 保存转换后的结果
        _write_json_file(output_json_path, labelme_structure)
        print(f"✓ SAHI转LabelMe完成！结果保存在: {output_json_path}")

    except Exception as e:
        print(f"✗ SAHI转LabelMe失败: {str(e)}")
        print(traceback.format_exc())  # 输出完整异常堆栈


def main():
    """主执行流程"""
    # 请根据你的实际情况修改这些路径
    
    # 原始数据路径（包含多边形标注的JSON文件）
    original_folder = "C:/Users/yangjinhua/Downloads/datasets/booth_images/"
    
    # 中间结果和最终结果的输出路径
    rotation_folder = "C:/Users/yangjinhua/Downloads/datasets/booth_rotated/"
    zero_angle_folder = "C:/Users/yangjinhua/Downloads/datasets/booth_zero_angle/"
    horizontal_folder = "C:/Users/yangjinhua/Downloads/datasets/booth_horizontal/"

    # ===== 根据需要选择执行以下步骤 =====
    
    # 步骤1: 将多边形转换为旋转框（如果你有polygon数据）
    # _convert_polygon_to_rotation(original_folder, rotation_folder)
    
    # 步骤2: 仅将旋转框的角度设为0（不改变顶点）
    # _zero_rotation_angles(rotation_folder, zero_angle_folder)
    
    # 步骤3: 转换为真正的水平矩形（会改变顶点）
    # _convert_to_horizontal_rect(rotation_folder, horizontal_folder)
    
    # 或者，如果你只想清零角度但保持顶点不变：
    # _zero_rotation_angles(rotation_folder, zero_angle_folder)
    
    # 步骤4: 将SAHI检测结果转换为LabelMe格式
    
    # 示例: 将角度清零（旋转坐标到0度）
    sahi_to_labelme_format(
        "D:/0-mingyang/code/mingyang_image_recognition/output/output_results/booth_obb_v13/booth/HongMu_sahi.json",
        "D:/0-mingyang/code/mingyang_image_recognition/output/test/HongMu.json",
        is_trun_zero=True,
    )


if __name__ == "__main__":
    main()