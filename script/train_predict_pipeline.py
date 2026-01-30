from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import re
import torch
from log_config import get_project_logger

# 导入train模块中的函数
from train import train_model, get_model_path, update_dataset_path

# 导入predict_sahi模块
from predict_sahi import start_predict

# 获取项目logger
logger = get_project_logger('train_predict_pipeline')

# 定义动态基础路径
project_dir = Path(__file__).resolve().parent.parent
logger.info(f"Project root directory: {project_dir}")

# ==================== 1. 自动化配置提取 ====================
# [配置项] 待训练的模型列表
# 脚本会自动根据模型名称识别版本（如 yolo11, yolov8, yolo26 等）并寻找预训练权重
models_to_train = [
    "yolov8s-obb.pt",
    "yolo11s-obb.pt",
    "yolo26s-obb.pt",
]

# [配置项] 数据集名称 (datasets 目录下的文件夹名)
dataset_name = "booth_seg"

# [配置项] 实验/轮次名称 (用于区分同一模型的不同训练配置)
exp_name = 'booth_obb_v1'

# [配置项] 预测图像路径列表 (支持单个或多个图片路径，或图片文件夹路径)
prediction_images = [
    # "/home/aistudio/YOLO/images/2024年展位图.jpg",
    # "/home/aistudio/YOLO/images/第十一届世界猪业博览会.jpeg",
    # "/home/aistudio/YOLO/images/长沙国际会展中心.jpg",
    # "/home/aistudio/YOLO/images/2020畜博会.png",

    f"{project_dir}/images",   # 使用目录
]

# ========================================================

def get_image_paths(image_sources):
    """
    从图片路径列表或文件夹路径列表中获取所有图片路径
    """
    image_paths = []
    for source in image_sources:
        source_path = Path(source)
        if source_path.is_file():
            # 如果是单个文件
            image_paths.append(str(source_path))
        elif source_path.is_dir():
            # 如果是目录，获取目录中所有图片文件（仅当前目录，避免递归到系统目录）
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            for ext in extensions:
                try:
                    image_paths.extend([str(p) for p in source_path.glob(ext)])
                except (NotADirectoryError, PermissionError, OSError):
                    # 忽略系统特殊文件或权限问题
                    continue
        else:
            logger.warning(f"Image source does not exist: {source}")
    
    return sorted(list(set(image_paths)))  # 去重并排序

def run_predict_sahi_after_training(model_path, image_path):
    """
    在训练完成后通过调用predict_sahi模块运行预测
    """
    try:
        logger.info(f"Starting prediction for image: {image_path}")
        
        # 调用predict_sahi模块的start_predict函数，传递数据集名称
        start_predict(model_path, image_path, dataset_name)
        
        logger.info("SAHI prediction completed successfully.")
    except Exception as e:
        logger.error(f"Error occurred when calling predict_sahi module: {str(e)}")


# ==================== 3. 主训练循环 ====================

# 定位数据集
dataset_root = project_dir / 'datasets' / dataset_name
dataset_yaml_path = dataset_root / 'dataset.yaml'

# 更新数据集路径配置
update_dataset_path(dataset_yaml_path, dataset_root)

# 存储所有训练完成的模型路径
trained_models = []

for model_filename in models_to_train:
    logger.info(f"{'='*50}")
    logger.info(f"Starting training for model: {model_filename}")
    logger.info(f"{'='*50}")

    # 获取预训练模型绝对路径
    yolo_model_path = get_model_path(model_filename, project_dir)
    if not yolo_model_path.exists():
        logger.warning(f"Pretrained model not found at {yolo_model_path}, will use auto-download.")

    # 执行模型训练
    best_model_path = train_model(yolo_model_path, dataset_yaml_path, project_dir, exp_name, dataset_name, epochs=3)

    if best_model_path is None:
        continue

    # 将训练完成的模型路径添加到列表中
    trained_models.append((best_model_path, model_filename))

# 所有模型训练完成后，统一进行预测
logger.info("="*50)
logger.info("All training completed. Starting unified prediction for all models and images...")
logger.info("="*50)

# 获取所有图片路径
all_image_paths = get_image_paths(prediction_images)
logger.info(f"Found {len(all_image_paths)} images to predict")

for model_path, model_filename in trained_models:
    logger.info(f"Running prediction for model: {model_filename}")
    for image_path in all_image_paths:
        run_predict_sahi_after_training(model_path, image_path)

logger.info("所有训练任务和预测任务已完成！")