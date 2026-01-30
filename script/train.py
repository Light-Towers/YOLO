from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import re
import sys
import torch
from log_config import get_project_logger

# 获取项目logger
logger = get_project_logger('train')

# ==================== 1. 自动化配置提取 ====================
# [配置项] 待训练的模型列表
# 脚本会自动根据模型名称识别版本（如 yolo11, yolov8, yolo26 等）并寻找预训练权重
models_to_train = [
    "yolo11m-obb.pt",
    # "yolov8s-obb.pt",
    # "yolo26n-obb.pt",
]

# [配置项] 数据集名称 (datasets 目录下的文件夹名)
dataset_name = "fixed_tiled_dataset_1"

# [配置项] 实验/轮次名称 (用于区分同一模型的不同训练配置)
exp_name = 'booth_obb_v1'

# [配置项] 预测图像路径列表 (支持单个或多个图片路径，或图片文件夹路径)
prediction_images = [
    "/home/aistudio/YOLO/images/2024年展位图.jpg",
    # "/home/aistudio/YOLO/images/第十一届世界猪业博览会.jpeg",
    # "/home/aistudio/YOLO/images/长沙国际会展中心.jpg",
    # "/home/aistudio/YOLO/images/2020畜博会.png",
    # "/home/aistudio/YOLO/images/2025年畜牧-展位分布图-1105-01.png"
]

# [动态检测] 硬件资源
device = "0" if torch.cuda.is_available() else "cpu"
# 动态计算工作线程：取 CPU 核心数的一半，最大不超过 8
workers = min(8, (os.cpu_count() or 1) // 2)
# ========================================================

# 2. 定义动态基础路径
project_dir = Path(__file__).resolve().parent.parent
logger.info(f"Project root directory: {project_dir}")
logger.info(f"Using device: {device}, workers: {workers}")

# 3. 辅助函数
def get_model_path(filename):
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

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # 确保 path 指向当前环境下的绝对路径
    data['path'] = str(Path(new_base_path).resolve())
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    logger.info(f"Updated dataset path in {yaml_path} to: {data['path']}")

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
            # 如果是目录，获取目录中所有图片文件
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            for ext in extensions:
                image_paths.extend([str(p) for p in source_path.glob(ext)])
                image_paths.extend([str(p) for p in source_path.rglob(f"*.{ext.lstrip('*.')}")])
        else:
            logger.warning(f"Image source does not exist: {source}")
    
    return sorted(list(set(image_paths)))  # 去重并排序

def run_predict_sahi_after_training(model_path, image_path):
    """
    在训练完成后通过调用predict_sahi模块运行预测
    """
    try:
        # 导入predict_sahi模块的start_predict函数
        from script.predict_sahi import start_predict
        
        logger.info(f"Starting prediction for image: {image_path}")
        
        # 调用predict_sahi模块的start_predict函数
        start_predict(model_path, image_path)
        
        logger.info("SAHI prediction completed successfully.")
    except ImportError:
        logger.error("Could not import predict_sahi module. Make sure predict_sahi.py is in the correct location.")
    except Exception as e:
        logger.error(f"Error occurred when calling predict_sahi module: {str(e)}")


# ==================== 4. 主训练循环 ====================

# 定位数据集
dataset_root = project_dir / 'datasets' / dataset_name
dataset_yaml_path = dataset_root / 'dataset.yaml'

# 更新数据集路径配置
update_dataset_path(dataset_yaml_path, dataset_root)

# 存储所有训练完成的模型路径和名称
trained_models = []

for model_filename in models_to_train:
    logger.info(f"{'='*50}")
    logger.info(f"Starting training for model: {model_filename}")
    logger.info(f"{'='*50}")

    # 获取预训练模型绝对路径
    yolo_model_path = get_model_path(model_filename)
    if not yolo_model_path.exists():
        logger.error(f"Pretrained model not found at {yolo_model_path}. Skipping...")
        continue

    # 定义简化的输出路径: output/models/{model_name}/{exp_name}/
    # 去掉 .pt 后缀作为文件夹名
    model_folder_name = Path(model_filename).stem
    train_save_dir = project_dir / 'output' / 'models' / model_folder_name

    # 加载模型
    model = YOLO(str(yolo_model_path))

    # 开始训练
    results = model.train(
        # 数据集配置文件
        data=str(dataset_yaml_path),
        
        epochs=300,                               # 训练轮数
        patience=50,                              # 早停耐心值
        imgsz=640,                                # 输入图像尺寸
        batch=0.9,                                # 【3种方式】16：固定方式；-1 自动计算最大可用batch； 0.8：按gpu内存分配
        device=device,                            # 训练设备（动态检测）
        workers=workers,                          # 工作线程数（动态检测）
        
        # ========== 项目相关参数 ==========
        project=str(train_save_dir),              # 指定模型训练输出根目录
        name=exp_name,                            # 指定实验名称
        save=True,                                # 保存训练结果和模型
        save_period=-1,                           # 仅在最后保存检查点
        pretrained=True,                          # 从预训练模型开始训练。可以是一个布尔值，也可以是加载权重的特定模型的字符串路径。增强训练效率和模型性能。
        
        # ========== 训练优化参数 ==========
        amp=True,                                 # 开启混合精度训练，某些显卡不需要
        cache=True,                               # 将数据集缓存到内存中 🚀
        compile=True,                             # 开启内核编译加速
        
        # ========== 关键修改3：调整数据增强策略 ==========
        # OBB任务对旋转敏感，需要谨慎调整旋转增强
        degrees=15.0,      # 【建议调低】展位图通常视角固定，避免过大的旋转
        translate=0.1,     # 平移增强
        scale=0.5,         # 缩放增强
        shear=0.0,         # 【建议关闭】剪切变换可能破坏旋转框的角度信息
        perspective=0.001, # 透视变换，保持较小的值
        flipud=0.0,        # 上下翻转【建议关闭】
        fliplr=0.5,        # 左右翻转可保留
        
        # 马赛克增强相关
        mosaic=1.0,        # 开启马赛克增强
        mixup=0.1,         # MixUp增强，不宜过高
        copy_paste=0.0,    # 【建议关闭】复制粘贴增强可能不适合OBB
        
        # ========== 关键修改4：OBB特定参数 ==========
        # YOLO OBB任务会自动处理旋转框，以下是可能需要关注的参数
        overlap_mask=False,  # 【注意】OBB任务不需要掩码重叠，应该设为False
        single_cls=True,     # 如果你的数据集中只有"展位"一个类别，设为True
        
        # ========== 优化器与学习率 ==========
        optimizer='auto',    # 自动选择优化器：[SGD, Adam, AdamW, NAdam, RAdam, RMSProp]
        lr0=0.01,           # 初始学习率
        lrf=0.01,           # 最终学习率系数 (lr0 * lrf)
        momentum=0.937,     # 动量
        weight_decay=0.0005, # 权重衰减
        warmup_epochs=3,    # 学习率预热轮数，有助于稳定训练初期
        warmup_momentum=0.8, # 预热期动量
        warmup_bias_lr=0.1, # 预热期偏置学习率
        
        # ========== 其他调整 ==========
        dropout=0.0,        # OBB任务通常不需要，防止小数据集过拟合
        cos_lr=True,        # 使用余弦退火学习率调度，可能帮助更好收敛
        # label_smoothing=0.0, # 标签平滑 (弃用)
        
        # ========== 验证相关参数 ==========
        val=True,           # 在训练期间进行验证
        plots=True,         # 在训练期间生成并保存图表
        resume=False,       # 是否从最近的检查点恢复训练
        
        # ========== 针对密集小目标的调整 ==========
        # 如果你的展位密集且较小，可以考虑以下调整
        # multi_scale=False,  # 多尺度训练（会增加训练时间）
        # nbs=64,             # 名义批量大小
        
        # ========== 调试参数 ==========
        verbose=True,       # 输出详细信息
        deterministic=True, # 确保可重复性
    )

    # 获取训练后的最佳模型路径
    best_model_path = train_save_dir / exp_name / 'weights' / 'best.pt'
    
    logger.info(f"Finished training for: {model_filename}")
    logger.info(f"Results saved in: {train_save_dir / exp_name}")

    # 将训练完成的模型路径添加到列表中
    trained_models.append((best_model_path, model_filename))

# 所有模型训练完成后，统一进行预测
logger.info("="*50)
logger.info("All training completed. Starting unified prediction for all models...")
logger.info("="*50)

# 获取所有图片路径
all_image_paths = get_image_paths(prediction_images)
logger.info(f"Found {len(all_image_paths)} images to predict")

for model_path, model_filename in trained_models:
    logger.info(f"Running prediction for model: {model_filename}")
    for image_path in all_image_paths:
        run_predict_sahi_after_training(model_path, image_path)

logger.info("\n所有训练任务和预测任务已完成！")