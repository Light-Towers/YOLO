from ultralytics import YOLO
import cv2
import numpy as np
import os

# 配置参数
model_path = "/home/aistudio/YOLO/models/train/booth_obb_v1/weights/best.pt"
source_image_path = "/home/aistudio/YOLO/images/第十一届世界猪业博览会.jpeg"

# 创建输出目录
output_dir = "/home/aistudio/YOLO/output_results/ultralytics_obb"
os.makedirs(output_dir, exist_ok=True)

# 从输入图像文件名生成输出文件名
input_filename = os.path.basename(source_image_path)
input_name, input_ext = os.path.splitext(input_filename)
output_path = os.path.join(output_dir, f"obb_{input_name}{input_ext}")

print(f"使用Ultralytics原生API进行OBB推理...")

# 1. 加载OBB模型（关键：指定task='obb'）
model = YOLO(model_path, task='obb')

# 2. 执行推理
results = model(source_image_path, conf=0.25, imgsz=640)

# 3. 处理结果
img = cv2.imread(source_image_path)
all_predictions = []

for result_idx, r in enumerate(results):
    # 检查是否有OBB结果
    if hasattr(r, 'obb') and r.obb is not None:
        # 获取旋转框的四个顶点（关键：xyxyxyxy格式）
        obb_boxes = r.obb.xyxyxyxy.cpu().numpy()
        confidences = r.obb.conf.cpu().numpy()
        
        print(f"找到 {len(obb_boxes)} 个旋转框")
        
        for i, (box, conf) in enumerate(zip(obb_boxes, confidences)):
            # box是4个点，每个点有(x, y)坐标
            points = box.reshape(4, 2).astype(int)
            
            print(f"物体 {i}: 旋转框顶点 = {points.tolist()}, 置信度 = {conf:.3f}")
            
            # 绘制旋转框的四个边
            for j in range(4):
                pt1 = tuple(points[j])
                pt2 = tuple(points[(j + 1) % 4])
                cv2.line(img, pt1, pt2, (0, 0, 255), 2)  # 红色边
            
            # 绘制顶点
            for pt in points:
                cv2.circle(img, tuple(pt), 5, (0, 255, 0), -1)  # 绿色顶点
            
            # 在框中心添加编号和置信度
            center = np.mean(points, axis=0).astype(int)
            label = f"{i}:{conf:.2f}"
            cv2.putText(img, label, tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            all_predictions.append({
                'id': i,
                'points': points,
                'confidence': conf,
                'bbox': [points[:, 0].min(), points[:, 1].min(), 
                        points[:, 0].max(), points[:, 1].max()]
            })

# 4. 保存结果
cv2.imwrite(output_path, img)
print(f"\n检测完成！共检测到 {len(all_predictions)} 个旋转框。")
print(f"结果已保存: {output_path}")

# 5. 输出详细信息
if all_predictions:
    print("\n旋转框详细信息（前5个）:")
    for pred in all_predictions[:5]:
        points = pred['points']
        print(f"  物体 {pred['id']}:")
        print(f"    置信度: {pred['confidence']:.3f}")
        print(f"    顶点坐标:")
        for j, pt in enumerate(points):
            print(f"      点{j+1}: ({pt[0]}, {pt[1]})")
        print(f"    水平外接框: ({pred['bbox'][0]}, {pred['bbox'][1]}, {pred['bbox'][2]}, {pred['bbox'][3]})")
else:
    print("警告：未检测到任何旋转框！")
    print("可能原因：")
    print("  1. 置信度阈值过高（当前conf=0.25）")
    print("  2. 模型训练不充分")
    print("  3. 图片中确实没有展位")