import os
import shutil
import random

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
    
    # 设置随机种子，保证每次分割结果一致
    random.seed(seed)
    
    # 创建输出目录
    train_img_dir = os.path.join(output_dir, 'images', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    train_label_dir = os.path.join(output_dir, 'labels', 'train')
    val_label_dir = os.path.join(output_dir, 'labels', 'val')
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 收集所有图片文件
    image_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            # 检查对应的txt文件是否存在
            base_name = os.path.splitext(file)[0]
            txt_file = base_name + '.txt'
            txt_path = os.path.join(input_dir, txt_file)
            
            if os.path.exists(txt_path):
                image_files.append((file, base_name, txt_file))
            else:
                print(f"警告: {file} 没有对应的标签文件，已跳过")
    
    print(f"找到 {len(image_files)} 对有效数据（图片+标签）")
    
    if len(image_files) == 0:
        print("错误：没有找到有效的数据文件！")
        return
    
    # 固定随机打乱
    random.shuffle(image_files)
    
    # 分割
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"\n分割结果（随机种子={seed}）:")
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    
    # 复制训练集
    for img_file, base_name, txt_file in train_files:
        # 复制图片
        shutil.copy2(
            os.path.join(input_dir, img_file),
            os.path.join(train_img_dir, img_file)
        )
        
        # 复制标签
        shutil.copy2(
            os.path.join(input_dir, txt_file),
            os.path.join(train_label_dir, txt_file)
        )
    
    # 复制验证集
    for img_file, base_name, txt_file in val_files:
        # 复制图片
        shutil.copy2(
            os.path.join(input_dir, img_file),
            os.path.join(val_img_dir, img_file)
        )
        
        # 复制标签
        shutil.copy2(
            os.path.join(input_dir, txt_file),
            os.path.join(val_label_dir, txt_file)
        )
    
    print(f"\n完成！文件已保存到: {output_dir}")
    

# 简单用法示例
if __name__ == "__main__":
    # 直接设置路径
    input_folder = "D:/0-mingyang/img_handle/YOLO_Train/yolo-labels"  # 修改为你的文件夹路径
    output_folder = "D:/Study/github/YOLO/datasets/booth_seg"

    # 调用函数
    split_dataset_fixed(input_folder, output_folder, train_ratio=0.8, seed=42)