from ultralytics import YOLO
import os
from pathlib import Path
import re
import torch
import sys

# 导入工程化工具
from src.utils import (
    get_device,
    get_project_root,
    get_logger,
    safe_mkdir,
    read_yaml,
    write_yaml,
)
from src.core import TRAINING_CONSTANTS, IMAGE_CONSTANTS

# 获取项目logger
logger = get_logger('train')

def train_model(model_path, dataset_yaml_path, project_dir, exp_name, dataset_name, epochs=300):
    """
    执行模型训练的核心函数
    """
    # 使用工具函数获取设备
    device = get_device()
    # 动态计算工作线程：取 CPU 核心数的一半，最大不超过 8
    workers = min(8, (os.cpu_count() or 1) // 2)

    # 定义简化的输出路径: output/models/{model_name}/{exp_name}/
    # 去掉 .pt 后缀作为文件夹名
    model_folder_name = Path(model_path).stem
    train_save_dir = project_dir / 'output' / 'models' / model_folder_name

    # 加载模型
    model = YOLO(str(model_path))

    # 开始训练
    results = model.train(
        # 数据集配置文件
        data=str(dataset_yaml_path),

        epochs=epochs,
        patience=TRAINING_CONSTANTS.DEFAULT_PATIENCE,
        imgsz=IMAGE_CONSTANTS.DEFAULT_IMAGE_SIZE,
        batch=TRAINING_CONSTANTS.DEFAULT_BATCH,
        device=device,
        workers=workers,

        # ========== 项目相关参数 ==========
        project=str(train_save_dir),
        name=exp_name,
        save=True,
        save_period=-1,
        pretrained=True,

        # ========== 数据增强策略 ==========
        degrees=TRAINING_CONSTANTS.DEFAULT_DEGREES,
        translate=TRAINING_CONSTANTS.DEFAULT_TRANSLATE,
        scale=TRAINING_CONSTANTS.DEFAULT_SCALE,
        shear=TRAINING_CONSTANTS.DEFAULT_SHEAR,
        perspective=TRAINING_CONSTANTS.DEFAULT_PERSPECTIVE,
        flipud=TRAINING_CONSTANTS.DEFAULT_FLIPUD,
        fliplr=TRAINING_CONSTANTS.DEFAULT_FLIPLR,
        mosaic=TRAINING_CONSTANTS.DEFAULT_MOSAIC,
        mixup=TRAINING_CONSTANTS.DEFAULT_MIXUP,
        copy_paste=TRAINING_CONSTANTS.DEFAULT_COPY_PASTE,

        # ========== OBB特定参数 ==========
        overlap_mask=False,
        single_cls=True,

        # ========== 优化器与学习率 ==========
        optimizer='auto',
        lr0=TRAINING_CONSTANTS.DEFAULT_LR,
        lrf=0.01,
        momentum=TRAINING_CONSTANTS.DEFAULT_MOMENTUM,
        weight_decay=TRAINING_CONSTANTS.DEFAULT_WEIGHT_DECAY,
        warmup_epochs=TRAINING_CONSTANTS.WARMUP_EPOCHS,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # ========== 其他调整 ==========
        dropout=0.0,
        cos_lr=True,

        # ========== 验证相关参数 ==========
        val=True,
        plots=True,
        resume=False,

        # ========== 调试参数 ==========
        verbose=True,
        deterministic=True,
    )

    # 获取训练后的最佳模型路径 - 使用实际的项目和实验名称
    actual_project_dir = train_save_dir / exp_name
    best_model_path = actual_project_dir / 'weights' / 'best.pt'
    
    # 如果上述路径不存在，尝试找到实际的输出目录
    if not best_model_path.exists():
        # 查找最新的训练输出目录
        exp_dirs = list(train_save_dir.glob(f"{exp_name}*"))
        if exp_dirs:
            # 按名称排序，取最后一个（最新的）
            latest_exp_dir = sorted(exp_dirs)[-1]
            best_model_path = latest_exp_dir / 'weights' / 'best.pt'
            logger.info(f"Found actual model path: {best_model_path}")
        else:
            logger.error(f"Could not find trained model at expected location: {actual_project_dir}")
            return None
    
    logger.info(f"Finished training for: {Path(model_path).name}")
    logger.info(f"Results saved in: {actual_project_dir}")
    
    return best_model_path

def get_model_path(filename, project_dir):
    """根据文件名自动定位预训练模型路径"""
    # 匹配版本号 (yolo11, yolov8, yolo26 等)，不区分大小写
    match = re.search(r'(yolo[a-z]*\d+)', filename.lower())
    version_dir = match.group(1) if match else ""
    return project_dir / 'models' / version_dir / filename

def update_dataset_path(yaml_path, new_base_path):
    """动态更新 dataset.yaml 中的 path 字段"""
    if not yaml_path.exists():
        logger.warning(f"{yaml_path} not found!")
        return

    # 使用工具函数读写 YAML
    data = read_yaml(yaml_path)
    data['path'] = str(Path(new_base_path).resolve())
    write_yaml(yaml_path, data)
    logger.info(f"Updated dataset path in {yaml_path} to: {data['path']}")

def main(model='yolov8s-obb.pt',
         dataset='fixed_tiled_dataset_1',
         exp_name='train_experiment',
         epochs=300,
         project_dir=None,
         update_dataset_path=False):
    """
    主函数：执行单个模型训练

    Args:
        model: 模型文件名 (如: yolov8s-obb.pt)
        dataset: 数据集名称 (datasets目录下的文件夹名)
        exp_name: 实验名称
        epochs: 训练轮数
        project_dir: 项目根目录路径，默认为脚本所在目录的上层目录
        update_dataset_path: 是否更新数据集yaml文件中的路径为绝对路径
    """
    # 设置项目目录
    if project_dir:
        project_dir = Path(project_dir).resolve()
    else:
        # 使用工具函数获取项目根目录
        project_dir = get_project_root()

    logger.info(f"项目根目录: {project_dir}")

    # 获取模型路径
    model_path = get_model_path(model, project_dir)
    if not model_path.exists():
        logger.warning(f"预训练模型在 {model_path} 未找到，将尝试自动下载。")
        # 如果模型不存在，使用构造的本地路径，让ultralytics库自动下载到该位置
    else:
        logger.info(f"找到本地模型: {model_path}")

    # 数据集配置
    dataset_root = project_dir / 'datasets' / dataset
    dataset_yaml_path = dataset_root / 'dataset.yaml'

    if not dataset_yaml_path.exists():
        logger.error(f"数据集配置文件不存在: {dataset_yaml_path}")
        logger.error(f"请确保数据集目录结构为: datasets/{dataset}/dataset.yaml")
        sys.exit(1)

    # 更新数据集路径（如果需要）
    if update_dataset_path:
        update_dataset_path(yaml_path=dataset_yaml_path, new_base_path=dataset_root)
    else:
        # 检查数据集路径是否正确，如果不正确则提示用户
        try:
            data = read_yaml(dataset_yaml_path)
            dataset_path = data.get('path', '')
            if dataset_path != str(dataset_root.resolve()):
                logger.warning(f"数据集路径可能不正确。当前: {dataset_path}")
                logger.warning(f"建议路径: {dataset_root.resolve()}")
                logger.warning(f"可将 update_dataset_path 参数设为 True 自动更新路径")
        except Exception as e:
            logger.warning(f"无法读取数据集配置文件: {e}")

    # 执行训练
    logger.info(f"开始训练模型: {model}")
    logger.info(f"数据集: {dataset}")
    logger.info(f"实验名称: {exp_name}")
    logger.info(f"训练轮数: {epochs}")

    best_model_path = train_model(
        model_path=str(model_path),
        dataset_yaml_path=dataset_yaml_path,
        project_dir=project_dir,
        exp_name=exp_name,
        dataset_name=dataset,
        epochs=epochs
    )

    if best_model_path:
        logger.info(f"训练完成！最佳模型保存位置: {best_model_path}")
    else:
        logger.error("训练失败！")
        sys.exit(1)


if __name__ == "__main__":
    # 在这里直接修改参数运行
    main(
        model='yolov8s-obb.pt',          # 模型文件名
        dataset='fixed_tiled_dataset_1',  # 数据集名称
        exp_name='train_experiment',      # 实验名称
        epochs=300,                       # 训练轮数
        update_dataset_path=False         # 是否更新数据集路径
    )