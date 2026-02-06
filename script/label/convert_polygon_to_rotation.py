import cv2
import numpy as np
from pathlib import Path

# 导入工程化工具
from src.utils import get_logger, safe_mkdir, read_json, write_json

logger = get_logger('label.convert_polygon_to_rotation')


def convert_polygon_to_rotation(input_folder, output_folder):
    """将指定文件夹中所有JSON文件里的多边形标注转换为旋转框标注。"""
    safe_mkdir(output_folder)
    logger.info(f"开始转换多边形 -> 旋转框: {input_folder} -> {output_folder}")

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    for json_file in input_folder.glob('*.json'):
        input_path = input_folder / json_file.name
        output_path = output_folder / json_file.name

        try:
            data = read_json(input_path)
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
            write_json(output_path, data, indent=2)
            logger.info(f"  已处理: {json_file.name}")

        except Exception as e:
            logger.error(f"  处理失败: {json_file.name} - {e}")

    logger.info(f"✓ 多边形转换完成！结果保存在: {output_folder}")


def zero_rotation_angles(input_folder, output_folder):
    """将指定文件夹中所有JSON文件里的旋转框标注的角度设置为0。"""
    safe_mkdir(output_folder)
    logger.info(f"开始清零旋转框角度: {input_folder} -> {output_folder}")

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    for json_file in input_folder.glob('*.json'):
        input_path = input_folder / json_file.name
        output_path = output_folder / json_file.name

        try:
            data = read_json(input_path)
            modified_count = 0

            for shape in data.get('shapes', []):
                if shape.get('shape_type') == 'rotation':
                    shape['direction'] = 0.0
                    modified_count += 1

            write_json(output_path, data, indent=2)
            logger.info(f"  已处理: {json_file.name} (修改了 {modified_count} 个旋转框)")

        except Exception as e:
            logger.error(f"  处理失败: {json_file.name} - {e}")

    logger.info(f"✓ 角度清零完成！结果保存在: {output_folder}")


def convert_to_horizontal_rect(input_folder, output_folder):
    """将旋转框转换为真正的水平矩形（修改顶点和角度）。"""
    safe_mkdir(output_folder)
    logger.info(f"开始转换为水平矩形: {input_folder} -> {output_folder}")

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    for json_file in input_folder.glob('*.json'):
        input_path = input_folder / json_file.name
        output_path = output_folder / json_file.name

        try:
            data = read_json(input_path)

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

            write_json(output_path, data, indent=2)
            logger.info(f"  已处理: {json_file.name}")

        except Exception as e:
            logger.error(f"  处理失败: {json_file.name} - {e}")

    logger.info(f"✓ 水平矩形转换完成！结果保存在: {output_folder}")


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
    logger.info(f"开始转换SAHI检测结果 -> LabelMe格式: {input_json_path} -> {output_json_path}")
    logger.info(f"  参数 - is_trun_zero: {is_trun_zero}")

    # 读取SAHI检测结果
    sahi_data = read_json(input_json_path)

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
            labelme_structure["imageHeight"] = int(max_y * 1.1)
            labelme_structure["imageWidth"] = int(max_x * 1.1)

    # 遍历每个检测结果
    for item in sahi_data:
        if "poly" not in item:
            continue

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
            # 不清零角度：直接使用最小外接矩形的4个顶点
            box = cv2.boxPoints(rect)
            box = box.reshape(-1, 2).tolist()
            direction_value = angle_rad

        # 创建shape对象
        shape = {
            "label": "booth",  # 统一标签为booth
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
    write_json(output_json_path, labelme_structure, indent=2)
    logger.info(f"✓ SAHI转LabelMe完成！结果保存在: {output_json_path}")


if __name__ == "__main__":
    from src.utils import get_project_root

    project_root = get_project_root()

    # 示例: 将SAHI检测结果转换为LabelMe格式
    sahi_to_labelme_format(
        input_json_path=project_root / 'output' / 'results' / 'booth_obb_v13' / 'booth' / 'HongMu_sahi.json',
        output_json_path=project_root / 'output' / 'test' / 'HongMu.json',
        is_trun_zero=False
    )
