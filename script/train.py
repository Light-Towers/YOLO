from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import re
import torch

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

# [动态检测] 硬件资源
device = "0" if torch.cuda.is_available() else "cpu"
# 动态计算工作线程：取 CPU 核心数的一半，最大不超过 8
workers = min(8, (os.cpu_count() or 1) // 2)
# ========================================================

# 2. 定义动态基础路径
project_dir = Path(__file__).resolve().parent.parent
print(f"Project root directory: {project_dir}")
print(f"Using device: {device}, workers: {workers}")

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
        print(f"Warning: {yaml_path} not found!")
        return

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # 确保 path 指向当前环境下的绝对路径
    data['path'] = str(Path(new_base_path).resolve())
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    print(f"Updated dataset path in {yaml_path} to: {data['path']}")

# ==================== 4. 主训练循环 ====================

# 定位数据集
dataset_root = project_dir / 'datasets' / dataset_name
dataset_yaml_path = dataset_root / 'dataset.yaml'

# 更新数据集路径配置
update_dataset_path(dataset_yaml_path, dataset_root)

for model_filename in models_to_train:
    print(f"\n{'='*30}")
    print(f"Starting training for model: {model_filename}")
    print(f"{'='*30}")

    # 获取预训练模型绝对路径
    yolo_model_path = get_model_path(model_filename)
    if not yolo_model_path.exists():
        print(f"Error: Pretrained model not found at {yolo_model_path}. Skipping...")
        continue

    # 定义简化的输出路径: output_results/models/{model_name}/{exp_name}/
    # 去掉 .pt 后缀作为文件夹名
    model_folder_name = Path(model_filename).stem
    train_save_dir = project_dir / 'output_results' / 'models' / model_folder_name
    
    # 加载模型
    model = YOLO(str(yolo_model_path))

    # 开始训练
    results = model.train(
        # 数据集配置文件
        data=str(dataset_yaml_path),
        
        epochs=300,                               # 训练轮数
        patience=50,                              # 早停耐心值
        imgsz=640,                                # 输入图像尺寸
        batch=0.9,                                # 自动分配 GPU 内存
        device=device,                            # 动态检测设备
        workers=workers,                          # 动态检测线程
        
        # ========== 项目相关参数 ==========
        project=str(train_save_dir),              # 输出主目录
        name=exp_name,                            # 实验子目录名
        save=True,                                # 保存训练结果和模型
        save_period=-1,                           # 仅在最后保存检查点
        pretrained=True,                          # 从预训练权重开始
        
        # ========== 训练优化参数 ==========
        amp=True,                                 # 混合精度训练
        cache=True,                               # 缓存数据集
        compile=True,                             # 内核编译加速
        
        # ========== 数据增强策略 (OBB 优化) ==========
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        
        # ========== OBB 特定参数 ==========
        overlap_mask=False,
        single_cls=True,
        
        # ========== 优化器与学习率 ==========
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # ========== 其他调整 ==========
        dropout=0.0,
        cos_lr=True,
        
        # ========== 验证与调试 ==========
        val=True,
        plots=True,
        resume=False,
        verbose=True,
        deterministic=True,
    )

    print(f"\nFinished training for: {model_filename}")
    print(f"Results saved in: {train_save_dir / exp_name}")

print("\n所有训练任务已完成！")
