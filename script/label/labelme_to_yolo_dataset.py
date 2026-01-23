import json
import os
import glob
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

def batch_refine_labelme_json(folder_path):
    # 支持的后缀名
    exts = ['.json']
    
    # 统计处理数量
    count = 0
    
    # 遍历文件夹
    for file_name in os.listdir(folder_path):
        if Path(file_name).suffix.lower() in exts:
            json_path = os.path.join(folder_path, file_name)
            
            try:
                # 1. 读取原始 JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                modified = False
                for shape in data['shapes']:
                    # 仅处理点数大于 2 的多边形 (手绘标注通常是 polygon)
                    if shape['shape_type'] == 'polygon' and len(shape['points']) > 2:
                        # 2. 转换为 numpy 格式供 OpenCV 计算
                        pts = np.array(shape['points'], dtype=np.float32)
                        
                        # 3. 计算最小外接矩形 (核心：支持任意角度倾斜)
                        # 返回值 rect: ((中心x, 中心y), (宽, 高), 旋转角度)
                        rect = cv2.minAreaRect(pts)
                        
                        # 4. 获取矩形的 4 个顶点坐标
                        box = cv2.boxPoints(rect)
                        
                        # 5. 更新数据 (将 4 个精准顶点回写)
                        shape['points'] = box.tolist()
                        modified = True
                
                # 6. 如果有改动，直接覆盖原文件
                if modified:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"已修正并覆盖: {file_name}")
                    count += 1
                    
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")

    print(f"\n处理完成！共修正了 {count} 个文件。")



def get_consistent_angle(shapes):
    """计算图中所有展位的众数角度，保证连排展位平行"""
    angles = []
    for shape in shapes:
        if shape['shape_type'] == 'polygon' and len(shape['points']) > 2:
            pts = np.array(shape['points'], dtype=np.float32)
            rect = cv2.minAreaRect(pts)
            angle = rect[2]
            # 归一化角度，处理 OpenCV 角度范围问题
            if rect[1][0] < rect[1][1]:
                angle = angle + 90
            angles.append(round(angle, 1)) # 保留一位小数求众数
    
    if not angles:
        return 0
    # 返回出现次数最多的角度
    return Counter(angles).most_common(1)[0][0]



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
    input_dir = "D:/0-mingyang/img_handle/YOLO_Train/labelme_labels" 
    batch_refine_labelme_json(input_dir)

    start_conversion(
        input_dir = input_dir,
        output_root="D:/0-mingyang/img_handle/YOLO_Train/booth_seg",
        class_list=["booth"]
    )