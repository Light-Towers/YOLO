#!/usr/bin/env python3
"""
将YOLO检测结果转换为标注数据格式
支持SAHI切割后的检测结果转换为labelme格式
"""

import json
import argparse
import math
from pathlib import Path


def rotate_point(px, py, cx, cy, angle_deg):
    """
    将点 (px, py) 绕中心点 (cx, cy) 旋转 angle_deg 度
    正角度：逆时针旋转
    负角度：顺时针旋转
    """
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # 平移到原点
    dx = px - cx
    dy = py - cy

    # 旋转
    new_x = dx * cos_a - dy * sin_a + cx
    new_y = dx * sin_a + dy * cos_a + cy

    return new_x, new_y


def calculate_rotation_angle(points):
    """
    计算旋转矩形的主轴角度（长边的角度）

    Args:
        points: 四个顶点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        旋转角度（度），范围 [-90, 90]
    """
    # 计算所有边的长度
    def distance(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    # 找到长边
    edges = [
        (points[0], points[1]),
        (points[1], points[2]),
        (points[2], points[3]),
        (points[3], points[0])
    ]

    edge_lengths = [distance(e[0], e[1]) for e in edges]
    max_idx = edge_lengths.index(max(edge_lengths))

    # 使用最长边计算角度
    p1, p2 = edges[max_idx]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # 计算角度（弧度转角度）
    angle = math.atan2(dy, dx) * 180 / math.pi

    # 将角度归一化到 [-90, 90]
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180

    return angle


def normalize_to_axis_aligned(points):
    """
    将旋转矩形的顶点转换为轴对齐（direction=0）的坐标
    即：反向旋转 -angle 度，得到水平/垂直的矩形

    Args:
        points: 原始旋转矩形的四个顶点

    Returns:
        (direction, normalized_points)
        - direction: 旋转角度
        - normalized_points: 轴对齐的四个顶点，按左上、右上、右下、左下排列
    """
    # 计算旋转角度
    direction = calculate_rotation_angle(points)

    # 计算中心点
    cx = sum(p[0] for p in points) / 4
    cy = sum(p[1] for p in points) / 4

    # 反向旋转所有顶点（旋转 -direction 度，使其轴对齐）
    rotated_points = []
    for p in points:
        new_x, new_y = rotate_point(p[0], p[1], cx, cy, -direction)
        rotated_points.append([new_x, new_y])

    # 按 labelme 格式排序：左上、右上、右下、左下
    # 先按 y 排序，再按 x 排序
    sorted_by_y = sorted(rotated_points, key=lambda p: p[1])
    top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])  # 上边两个，按 x 排序
    bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])  # 下边两个，按 x 排序

    # 左上、右上、右下、左下
    normalized = [top_two[0], top_two[1], bottom_two[1], bottom_two[0]]

    return direction, normalized


def find_matching_image(detection_file_name, images_dir):
    """
    根据检测结果文件名查找匹配的图片文件

    Args:
        detection_file_name: 检测结果文件名（如 "2011-长沙孕婴童产业_sahi.json"）
        images_dir: 图片目录路径

    Returns:
        匹配的图片文件名，如果未找到则返回空字符串
    """
    images_path = Path(images_dir)

    # 移除 _sahi.json 后缀，得到基础名称
    base_name = detection_file_name.replace("_sahi.json", "").replace(".json", "")

    # 在 images 目录中查找匹配的图片
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        image_file = images_path / f"{base_name}{ext}"
        if image_file.exists():
            return image_file.name

    return ""


def convert_detection_to_annotation(detection_file, output_file, image_width=1920, image_height=800, images_dir=None):
    """
    转换检测文件到标注文件格式

    Args:
        detection_file: 检测结果JSON文件路径
        output_file: 输出标注JSON文件路径
        image_width: 图像宽度（默认1920）
        image_height: 图像高度（默认800）
        images_dir: 图片目录路径（用于查找匹配的图片）
    """
    # 读取检测结果
    with open(detection_file, 'r', encoding='utf-8') as f:
        detections = json.load(f)

    # 查找匹配的图片
    detection_name = Path(detection_file).name
    image_path = ""
    if images_dir:
        image_path = find_matching_image(detection_name, images_dir)

    # 构建标注数据结构（labelme格式）
    annotation_data = {
        "version": "5.10.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    # 遍历检测结果，转换为标注格式
    for det in detections:
        # 从poly获取四个顶点坐标
        poly = det['poly']  # [x1, y1, x2, y2, x3, y3, x4, y4]

        # 转换为labelme的points格式
        raw_points = [
            [poly[0], poly[1]],
            [poly[2], poly[3]],
            [poly[4], poly[5]],
            [poly[6], poly[7]]
        ]

        # 计算旋转角度并转换为轴对齐坐标
        direction, points = normalize_to_axis_aligned(raw_points)

        # 创建标注对象
        shape = {
            "label": "booth",  # 默认使用booth作为标签名
            "points": points,
            "shape_type": "rotation",  # YOLO-OBB 旋转边界框
            "direction": 0.0,  # 设置为0，points已是轴对齐坐标
            "group_id": None,
            "difficult": False,
            "flags": {},
            "attributes": {}
        }

        annotation_data['shapes'].append(shape)

    # 保存标注文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, f, indent=2, ensure_ascii=False)

    print(f"转换完成!")
    print(f"输入文件: {detection_file}")
    print(f"输出文件: {output_file}")
    print(f"图片路径: {image_path}")
    print(f"共转换 {len(annotation_data['shapes'])} 个标注对象")


def batch_convert(input_dir, output_dir, image_width=1920, image_height=800, images_dir=None):
    """
    批量转换目录下所有检测文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        image_width: 图像宽度
        image_height: 图像高度
        images_dir: 图片目录路径
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有_sahi.json文件
    detection_files = list(input_path.glob("*_sahi.json"))

    if not detection_files:
        print(f"未找到 *_sahi.json 文件在 {input_dir}")
        return

    print(f"找到 {len(detection_files)} 个检测文件")

    for detection_file in detection_files:
        # 生成输出文件名
        output_file = output_path / detection_file.name.replace("_sahi.json", ".json")

        # 转换单个文件
        convert_detection_to_annotation(
            detection_file,
            output_file,
            image_width,
            image_height,
            images_dir
        )


def main():
    parser = argparse.ArgumentParser(description='将YOLO检测结果转换为labelme标注格式')
    parser.add_argument('input', type=str, help='输入文件或目录')
    parser.add_argument('output', type=str, help='输出文件或目录')
    parser.add_argument('--width', type=int, default=1920, help='图像宽度（默认1920）')
    parser.add_argument('--height', type=int, default=800, help='图像高度（默认800）')
    parser.add_argument('--images_dir', type=str, default=None, help='图片目录路径（用于设置imagePath）')
    parser.add_argument('--batch', action='store_true', help='批量转换模式')

    args = parser.parse_args()

    if args.batch:
        batch_convert(args.input, args.output, args.width, args.height, args.images_dir)
    else:
        convert_detection_to_annotation(args.input, args.output, args.width, args.height, args.images_dir)


if __name__ == '__main__':
    # 示例用法
    # 单文件转换:
    # python convert_detection_to_annotation.py input.json output.json

    # 批量转换:
    # python convert_detection_to_annotation.py input_dir output_dir --batch

    # 带图像尺寸:
    # python convert_detection_to_annotation.py input.json output.json --width 1920 --height 800

    main()
