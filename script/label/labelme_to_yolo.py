import json
import os
import glob
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

def labelme_to_yolo_seg(input_dir, output_dir, class_list):
    """
    将 Labelme JSON 转换为 YOLO 实例分割格式 (.txt)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_w = data['imageWidth']
        img_h = data['imageHeight']
        img_name = os.path.basename(json_path).replace(".json", "")
        
        yolo_lines = []
        for shape in data['shapes']:
            label = shape['label']
            if label not in class_list:
                continue
            
            cls_id = class_list.index(label)
            points = shape['points']
            
            # 归一化坐标并展平
            # YOLO 分割格式需要: cls x1 y1 x2 y2 ...
            normalized_points = []
            for p in points:
                norm_x = p[0] / img_w
                norm_y = p[1] / img_h
                normalized_points.append(f"{norm_x:.6f} {norm_y:.6f}")
            
            yolo_lines.append(f"{cls_id} {' '.join(normalized_points)}")
        
        # 写入 TXT 文件
        with open(os.path.join(output_dir, f"{img_name}.txt"), 'w') as f_out:
            f_out.write("\n".join(yolo_lines))

    print(f"转换完成！共处理 {len(json_files)} 个文件。")




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

# def batch_refine_labelme_json(folder_path):
#     exts = ['.json']
#     count = 0
    
#     for file_name in os.listdir(folder_path):
#         if Path(file_name).suffix.lower() in exts:
#             json_path = os.path.join(folder_path, file_name)
            
#             try:
#                 with open(json_path, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
                
#                 # 1. 获取本张图中统一的角度，保证连排展位平行
#                 common_angle = get_consistent_angle(data['shapes'])
                
#                 modified = False
#                 for shape in data['shapes']:
#                     if shape['shape_type'] == 'polygon' and len(shape['points']) > 2:
#                         pts = np.array(shape['points'], dtype=np.float32)
#                         rect = cv2.minAreaRect(pts)
#                         center, size = rect[0], rect[1]
                        
#                         # 2. 使用统一角度重新构建矩形，消除连排展位的角度偏差
#                         fixed_rect = (center, size, common_angle)
#                         box = cv2.boxPoints(fixed_rect)
                        
#                         # 3. 顶点排序：左上、右上、右下、左下 (保证相邻展位点序一致)
#                         # 按 y 坐标排序找到上方两个点，再按 x 排序
#                         box = sorted(box, key=lambda x: x[1])
#                         top_pts = sorted(box[:2], key=lambda x: x[0])
#                         bottom_pts = sorted(box[2:], key=lambda x: x[0], reverse=True)
#                         sorted_box = np.array([top_pts[0], top_pts[1], bottom_pts[0], bottom_pts[1]])

#                         shape['points'] = sorted_box.tolist()
#                         modified = True
                
#                 if modified:
#                     with open(json_path, 'w', encoding='utf-8') as f:
#                         json.dump(data, f, ensure_ascii=False, indent=2)
#                     print(f"已对齐并覆盖: {file_name} (统一角度: {common_angle}°)")
#                     count += 1
                    
#             except Exception as e:
#                 print(f"处理 {file_name} 出错: {e}")

#     print(f"\n处理完成！共对齐了 {count} 个文件。")



if __name__ == "__main__":
    # # --- 配置参数 ---
    # # 1. 填入你的类别名称，顺序必须固定
    # my_classes = ["booth"] 
    # # 2. 你的标注 JSON 文件夹路径
    # input_jsons = "D:/0-mingyang/img_handle/YOLO_Train/labelme_labels" 
    # # 3. 转换后 TXT 的保存路径
    # output_labels = "D:/0-mingyang/img_handle/YOLO_Train/yolo_labels"

    # labelme_to_yolo_seg(input_jsons, output_labels, my_classes)


    input_jsons = "D:/0-mingyang/img_handle/YOLO_Train/labelme_labels" 
    batch_refine_labelme_json(input_jsons)