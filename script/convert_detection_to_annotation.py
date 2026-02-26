#!/usr/bin/env python3
"""
将YOLO检测结果转换为标注数据格式
支持SAHI切割后的检测结果转换为labelme格式
"""

import json
import argparse
from pathlib import Path


def convert_detection_to_annotation(detection_file, output_file, image_width=1920, image_height=800):
    """
    转换检测文件到标注文件格式

    Args:
        detection_file: 检测结果JSON文件路径
        output_file: 输出标注JSON文件路径
        image_width: 图像宽度（默认1920）
        image_height: 图像高度（默认800）
    """
    # 读取检测结果
    with open(detection_file, 'r', encoding='utf-8') as f:
        detections = json.load(f)

    # 构建标注数据结构（labelme格式）
    annotation_data = {
        "version": "5.10.1",
        "flags": {},
        "shapes": [],
        "imagePath": "",
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    # 遍历检测结果，转换为标注格式
    for det in detections:
        # 从poly获取四个顶点坐标
        poly = det['poly']  # [x1, y1, x2, y2, x3, y3, x4, y4]

        # 转换为labelme的points格式
        points = [
            [poly[0], poly[1]],
            [poly[2], poly[3]],
            [poly[4], poly[5]],
            [poly[6], poly[7]]
        ]

        # 创建标注对象
        shape = {
            "label": "booth",  # 默认使用booth作为标签名
            "points": points,
            "shape_type": "polygon",  # 使用polygon而不是rotation
            "direction": 0.0,
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
    print(f"共转换 {len(annotation_data['shapes'])} 个标注对象")


def batch_convert(input_dir, output_dir, image_width=1920, image_height=800):
    """
    批量转换目录下所有检测文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        image_width: 图像宽度
        image_height: 图像高度
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
            image_height
        )


def main():
    parser = argparse.ArgumentParser(description='将YOLO检测结果转换为labelme标注格式')
    parser.add_argument('input', type=str, help='输入文件或目录')
    parser.add_argument('output', type=str, help='输出文件或目录')
    parser.add_argument('--width', type=int, default=1920, help='图像宽度（默认1920）')
    parser.add_argument('--height', type=int, default=800, help='图像高度（默认800）')
    parser.add_argument('--batch', action='store_true', help='批量转换模式')

    args = parser.parse_args()

    if args.batch:
        batch_convert(args.input, args.output, args.width, args.height)
    else:
        convert_detection_to_annotation(args.input, args.output, args.width, args.height)


if __name__ == '__main__':
    # 示例用法
    # 单文件转换:
    # python convert_detection_to_annotation.py input.json output.json

    # 批量转换:
    # python convert_detection_to_annotation.py input_dir output_dir --batch

    # 带图像尺寸:
    # python convert_detection_to_annotation.py input.json output.json --width 1920 --height 800

    main()
