import cv2
import numpy as np
from pathlib import Path

# 导入工程化工具
from src.utils import get_logger, safe_mkdir, read_json, write_json

logger = get_logger('label.label_format_utils')


def _read_json_file(file_path):
    """读取JSON文件，添加异常处理"""
    try:
        return read_json(file_path)
    except Exception as e:
        logger.error(f"错误：读取文件失败 - {file_path}, 错误: {str(e)}")
        raise


def _write_json_file(file_path, data):
    """写入JSON文件，添加异常处理"""
    try:
        safe_mkdir(Path(file_path).parent)
        write_json(file_path, data, indent=2)
    except Exception as e:
        logger.error(f"错误：写入文件失败 - {file_path}, 错误: {str(e)}")
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
    if json_file_path and Path(json_file_path).exists():
        # 生成可能的图像文件路径（去除.json扩展名，替换为常见图像格式）
        base_path = Path(json_file_path).stem
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        for ext in image_extensions:
            image_path = Path(json_file_path).parent / f"{base_path}{ext}"
            if image_path.exists():
                try:
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        height, width = image.shape[:2]
                        return width, height
                except Exception as e:
                    logger.warning(f"警告：无法读取图像文件 {image_path}: {str(e)}")
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
    safe_mkdir(output_folder)
    logger.info(f"开始转换多边形 -> 旋转框: {input_folder} -> {output_folder}")

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    for json_file in input_folder.glob('*.json'):
        input_path = input_folder / json_file.name
        output_path = output_folder / json_file.name

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
            logger.info(f"  已处理: {json_file.name}")

        except Exception as e:
            logger.error(f"  跳过: {json_file.name} (处理失败: {e})")
            continue

    logger.info(f"✓ 多边形转换完成！结果保存在: {output_folder}")


def _zero_rotation_angles(input_folder, output_folder):
    """将旋转框角度设为0"""
    safe_mkdir(output_folder)
    logger.info(f"开始清零旋转框角度: {input_folder} -> {output_folder}")

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    for json_file in input_folder.glob('*.json'):
        input_path = input_folder / json_file.name
        output_path = output_folder / json_file.name

        try:
            data = _read_json_file(input_path)
            modified_count = 0

            for shape in data.get('shapes', []):
                if shape.get('shape_type') == 'rotation':
                    shape['direction'] = 0.0
                    modified_count += 1

            _write_json_file(output_path, data)
            logger.info(f"  已处理: {json_file.name} (修改了 {modified_count} 个旋转框)")

        except Exception as e:
            logger.error(f"  跳过: {json_file.name} (处理失败: {e})")
            continue

    logger.info(f"✓ 角度清零完成！结果保存在: {output_folder}")


def _convert_to_horizontal_rect(input_folder, output_folder):
    """将旋转框转换为水平矩形"""
    safe_mkdir(output_folder)
    logger.info(f"开始转换为水平矩形: {input_folder} -> {output_folder}")

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    for json_file in input_folder.glob('*.json'):
        input_path = input_folder / json_file.name
        output_path = output_folder / json_file.name

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
            logger.info(f"  已处理: {json_file.name}")

        except Exception as e:
            logger.error(f"  跳过: {json_file.name} (处理失败: {e})")
            continue

    logger.info(f"✓ 水平矩形转换完成！结果保存在: {output_folder}")


def sahi_to_labelme_format(input_json_path, output_json_path, is_trun_zero=False):
    """
    将SAHI检测结果JSON文件转换为LabelMe格式的JSON文件

    参数:
        - input_json_path: 输入的SAHI JSON文件路径
        - output_json_path: 输出的LabelMe JSON文件路径
        - is_trun_zero: 是否将角度清零（True时旋转坐标到0度，False时保留原始角度）
    """
    logger.info(f"开始转换SAHI检测结果 -> LabelMe格式: {input_json_path} -> {output_json_path}")
    logger.info(f"  参数 - is_trun_zero: {is_trun_zero}")

    # 读取SAHI检测结果
    sahi_data = _read_json_file(input_json_path)

    # 创建LabelMe格式的基本结构
    labelme_structure = {
        "version": "3.3.6",
        "flags": {},
        "shapes": [],
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
            # 【修复逻辑】：不旋转坐标系，而是直接提取该多边形的水平最大边界
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
    logger.info(f"✓ SAHI转LabelMe完成！结果保存在: {output_json_path}")


if __name__ == "__main__":
    from src.utils import get_project_root

    project_root = get_project_root()

    # 示例: 将角度清零（旋转坐标到0度）
    sahi_to_labelme_format(
        input_json_path=project_root / 'output' / 'results' / 'booth_obb_v13' / 'booth' / 'HongMu_sahi.json',
        output_json_path=project_root / 'output' / 'test' / 'HongMu.json',
        is_trun_zero=True,
    )
