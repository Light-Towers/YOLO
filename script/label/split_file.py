import os
import random
import shutil
from pathlib import Path

from src.core import DATASET_CONSTANTS

# 导入工程化工具
from src.utils import (
    get_image_files,
    get_logger,
    get_project_root,
    safe_mkdir,
)

# 获取项目logger
logger = get_logger('label.split_file')

def split_dataset_fixed(input_dir, output_dir, train_ratio=0.8, seed=42):
    """
    固定分割数据集，目录结构为：
    output_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """
    # 使用常量中的默认值
    train_ratio = train_ratio if train_ratio != 0.8 else DATASET_CONSTANTS.DEFAULT_TRAIN_RATIO

    # 设置随机种子，保证每次分割结果一致
    random.seed(seed)

    # 使用工具函数创建输出目录
    train_img_dir = Path(output_dir) / 'images' / 'train'
    val_img_dir = Path(output_dir) / 'images' / 'val'
    train_label_dir = Path(output_dir) / 'labels' / 'train'
    val_label_dir = Path(output_dir) / 'labels' / 'val'

    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        safe_mkdir(dir_path)

    # 使用工具函数收集图片文件
    input_path = Path(input_dir)
    image_files = []
    for img_file in get_image_files(input_path, recursive=False):
        # 检查对应的txt文件是否存在
        base_name = img_file.stem
        txt_file = img_file.with_suffix('.txt')

        if txt_file.exists():
            image_files.append((img_file.name, base_name, str(txt_file)))
        else:
            logger.warning(f"警告: {img_file.name} 没有对应的标签文件，已跳过")
    
    logger.info(f"找到 {len(image_files)} 对有效数据（图片+标签）")
    
    if len(image_files) == 0:
        logger.error("错误：没有找到有效的数据文件！")
        return
    
    # 固定随机打乱
    random.shuffle(image_files)
    
    # 分割
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    logger.info(f"\n分割结果（随机种子={seed}）:")
    logger.info(f"训练集: {len(train_files)} 个文件")
    logger.info(f"验证集: {len(val_files)} 个文件")
    
    # 复制训练集
    input_path = Path(input_dir)
    for img_file, base_name, txt_file in train_files:
        # 复制图片
        shutil.copy2(
            input_path / img_file,
            train_img_dir / img_file
        )

        # 复制标签
        shutil.copy2(
            Path(txt_file),
            train_label_dir / Path(txt_file).name
        )

    # 复制验证集
    for img_file, base_name, txt_file in val_files:
        # 复制图片
        shutil.copy2(
            input_path / img_file,
            val_img_dir / img_file
        )

        # 复制标签
        shutil.copy2(
            Path(txt_file),
            val_label_dir / Path(txt_file).name
        )
    
    logger.info(f"\n完成！文件已保存到: {output_dir}")
    

# 简单用法示例
if __name__ == "__main__":
    project_root = get_project_root()

    # 直接设置路径
    input_folder = str(project_root / "datasets" / "source" / "yolo-labels")
    output_folder = str(project_root / "datasets" / "booth_seg")

    # 调用函数
    split_dataset_fixed(input_folder, output_folder, train_ratio=0.8, seed=42)