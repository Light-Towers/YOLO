from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path

# 导入工程化工具
from src.utils import (
    get_device,
    get_project_root,
    get_logger,
    safe_mkdir,
    write_json,
)
from src.core import INFERENCE_CONSTANTS

# 获取项目logger
logger = get_logger('predict_obb')


def start_predict(model_path, image_path, dataset_name=None, output_dir=None, model_name=None):
    """使用Ultralytics原生API进行OBB推理"""
    # 确保必须提供model_path和image_path参数
    if model_path is None:
        raise ValueError("model_path must be provided")
    if image_path is None:
        raise ValueError("image_path must be provided")

    model_path = str(model_path)
    image_path = str(image_path)

    # 如果没有显式提供model_name，则从模型路径中提取
    if model_name is None:
        model_name = Path(model_path).parent.parent.name
        if model_name == "weights":
            model_name = Path(model_path).parent.parent.parent.name
        if model_name in ["weights", "models", "results"]:
            model_file_name = Path(model_path).stem
            parts = model_file_name.split('_')
            if len(parts) > 1:
                model_name = '_'.join(parts[:-1])
            else:
                model_name = model_file_name
    else:
        model_name = str(model_name)

    # 构建输出目录结构
    project_dir = get_project_root()
    if output_dir is not None:
        base_output_dir = Path(output_dir)
    else:
        base_output_dir = project_dir / "output" / "results"

    # 如果提供了数据集名称，使用它，否则尝试从模型路径推断
    if dataset_name is None:
        parent_parts = str(Path(model_path).parent)
        if "booth" in parent_parts:
            dataset_name = "booth"
        else:
            dataset_name = "default_dataset"
    else:
        dataset_name = str(dataset_name)

    # 创建模型和数据集的输出目录
    model_dataset_dir = base_output_dir / model_name / dataset_name
    safe_mkdir(model_dataset_dir)

    # 从输入图像文件名生成输出文件名
    input_filename = os.path.basename(image_path)
    input_name, input_ext = os.path.splitext(input_filename)
    output_filename = f"obb_{input_name}"
    output_path = model_dataset_dir / f"{output_filename}{input_ext}"

    logger.info(f"模型名称: {model_name}")
    logger.info(f"数据集名称: {dataset_name}")
    logger.info(f"输出目录: {model_dataset_dir}")
    logger.info(f"输出文件: {output_path}")

    # 1. 加载OBB模型
    model = YOLO(model_path, task='obb')
    model.to(get_device())

    # 2. 执行推理
    logger.info("正在进行OBB推理，请稍候...")
    results = model.predict(
        source=image_path,
        save=False,
        conf=INFERENCE_CONSTANTS.DEFAULT_CONFIDENCE,
        iou=INFERENCE_CONSTANTS.DEFAULT_IOU,
        device=get_device(),
        exist_ok=True,
    )

    # 3. 处理结果
    img = cv2.imread(image_path)
    all_predictions = []

    for result_idx, r in enumerate(results):
        if hasattr(r, 'obb') and r.obb is not None:
            obb_boxes = r.obb.xyxyxyxy.cpu().numpy()
            confidences = r.obb.conf.cpu().numpy()

            logger.info(f"找到 {len(obb_boxes)} 个旋转框")

            for i, (box, conf) in enumerate(zip(obb_boxes, confidences)):
                points = box.reshape(4, 2).astype(int)

                # 绘制旋转框的四个边
                for j in range(4):
                    pt1 = tuple(points[j])
                    pt2 = tuple(points[(j + 1) % 4])
                    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

                # 绘制顶点
                for pt in points:
                    cv2.circle(img, tuple(pt), 5, (0, 255, 0), -1)

                all_predictions.append({
                    'id': i,
                    'points': points.tolist(),
                    'confidence': float(conf),
                    'bbox': [float(points[:, 0].min()), float(points[:, 1].min()),
                            float(points[:, 0].max()), float(points[:, 1].max())]
                })

    # 4. 保存结果
    cv2.imwrite(str(output_path), img)
    logger.info(f"检测完成！共检测到 {len(all_predictions)} 个旋转框。")
    logger.info(f"结果已保存: {output_path}")

    # 5. 保存JSON结果
    json_path = model_dataset_dir / f"{output_filename}.json"
    write_json(json_path, all_predictions, indent=4)

    return all_predictions


if __name__ == "__main__":
    project_root = get_project_root()

    model_path = str(project_root / 'output' / 'models' / 'train' / 'booth_obb_v13' / 'weights' / 'best.pt')
    image_path = str(project_root / 'images' / '2024-畜博会.jpg')
    dataset_name = None
    output_dir = None

    start_predict(model_path, image_path, dataset_name, output_dir)
