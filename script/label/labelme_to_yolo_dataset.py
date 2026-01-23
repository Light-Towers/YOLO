import json
import os
import glob
import shutil
import random

# --- 1. 基础工具逻辑：负责文件夹 ---
def prepare_structure(root_path):
    """只负责创建文件夹结构"""
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(root_path, folder), exist_ok=True)

# --- 2. 基础工具逻辑：负责数据分配 ---
def get_split_files(input_dir, train_ratio):
    """只负责计算哪些文件该去 train，哪些该去 val"""
    all_jsons = glob.glob(os.path.join(input_dir, "*.json"))
    random.seed(42)
    random.shuffle(all_jsons)
    
    index = int(len(all_jsons) * train_ratio)
    return all_jsons[:index], all_jsons[index:]

# --- 3. 核心转换逻辑：只负责解析 JSON 文本 ---
def convert_to_yolo_format(json_data, class_list):
    """只负责把 json 对象里的坐标变成 yolo 字符串列表"""
    w, h = json_data['imageWidth'], json_data['imageHeight']
    results = []
    
    for shape in json_data['shapes']:
        label = shape['label']
        if label not in class_list:
            continue
        
        cls_id = class_list.index(label)
        points = shape['points']
        # 核心数学转换
        norm_pts = [f"{p[0]/w:.6f} {p[1]/h:.6f}" for p in points]
        results.append(f"{cls_id} {' '.join(norm_pts)}")
    
    return results

# --- 4. 核心文件逻辑：负责找图片并复制 ---
def copy_matched_image(json_path, image_name_in_json, target_dir):
    """只负责把图片找到并复制到指定目录，支持覆盖"""
    img_dir = os.path.dirname(json_path)
    # 尝试多种匹配方式
    possible_names = [image_name_in_json, os.path.basename(json_path).replace(".json", ".jpg"), 
                      os.path.basename(json_path).replace(".json", ".png")]
    
    for name in possible_names:
        src = os.path.join(img_dir, name)
        if os.path.exists(src) and not os.path.isdir(src):
            shutil.copy2(src, os.path.join(target_dir, os.path.basename(src)))
            return True
    return False

# --- 5. 顶层业务逻辑：把上面的积木搭起来 ---
def start_conversion(input_dir, output_root, class_list, ratio=0.8):
    # 第一步：盖房子（建目录）
    prepare_structure(output_root)
    
    # 第二步：分猪肉（分名单）
    train_files, val_files = get_split_files(input_dir, ratio)
    
    # 第三步：干活（处理文件）
    dataset_tasks = [('train', train_files), ('val', val_files)]
    
    for mode, file_list in dataset_tasks:
        for json_path in file_list:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 写标签文件
            yolo_lines = convert_to_yolo_format(data, class_list)
            label_name = os.path.basename(json_path).replace(".json", ".txt")
            with open(os.path.join(output_root, 'labels', mode, label_name), 'w') as f_out:
                f_out.write("\n".join(yolo_lines))
            
            # 搬运图片
            img_target_dir = os.path.join(output_root, 'images', mode)
            copy_matched_image(json_path, data.get('imagePath', ''), img_target_dir)

    print("转换及分发完成！")

# --- 运行处 ---
if __name__ == "__main__":
    start_conversion(
        input_dir="D:/0-mingyang/img_handle/YOLO_Train/labelme_labels",
        output_root="D:/0-mingyang/img_handle/YOLO_Train/booth_seg",
        class_list=["booth"]
    )