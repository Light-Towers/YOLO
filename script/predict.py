from ultralytics import YOLO
import cv2
import os

# 1. 加载预训练模型
# yolov8x-worldv2.pt  加载 YOLO-World 模型（推荐使用 v8s 或 v8m，权衡速度与精度）

# model_path="/workspace/models/yolo11/yolo11n.pt"
model_path="/home/aistudio/YOLO/models/train/booth_seg_v12/weights/best.pt"

model = YOLO(model_path)

# 2. 推理
# source_image = "/workspace/image/长沙国际会展中心.jpg"
# source_image = "/workspace/image/2025年畜牧-展位分布图-1105-01.png"
# source_image = "/workspace/image/第十一届世界猪业博览会.jpeg"
source_image = "/home/aistudio/YOLO/images/2024年展位图-test.png"
# source_image = "/workspace/image/2024年展位图_压缩.jpg"
# source_image = "/workspace/image/2024年展位图.jpg"
results = model.predict(
    source=source_image, 
    save=False,                     # 关闭自动保存绘图
    conf=0.7,                         # 置信度
    iou=0.4,                          # 稍低于默认值，避免密集展位被误删
    # stream=True,                      # 逐帧处理，不一次性把所有结果塞进显存
    device=0,                          # 指定使用 GPU (0 表示第一块显卡)
    project="/home/aistudio/YOLO/output_results",   # 指定项目根目录
    name="booth_segment",            # 指定实验名称
    exist_ok=True,                     # 覆盖已有目录
    imgsz=(4051,4286),                  # 提高输入分辨率，有助于检测图中密集的小展位
)

# 3. 读取图像并绘制红色框
img = cv2.imread(source_image)

# if results and len(results) > 0:
#     for box in results[0].boxes:
#         # 解析结果并打印坐标
#         coords = box.xyxy[0].tolist() # 得到左上角和右下角坐标
#         conf = box.conf[0].item()
#         print(f"位置: {coords}, 置信度: {conf:.4f}")
        
#         # 绘制红色框
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)



# 如果是分割模型，results[0] 会包含 masks 属性
if results[0].masks is not None:
    for mask in results[0].masks:
        # 获取多边形坐标 (Polygons)
        segments = mask.xy  # 这是一个列表，包含多边形的顶点坐标
        
        for segment in segments:
            # 将坐标转换为整数并调整形状以适应 cv2.polylines
            import numpy as np
            pts = np.array(segment, np.int32).reshape((-1, 1, 2))
            # 绘制绿色多边形轮廓
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        

# 4. 保存结果
output_path = "/home/aistudio/YOLO/output_results/red_boxes.jpg"
cv2.imwrite(output_path, img)
print(f"红色框标注完成，保存至: {output_path}")
print(f"检测到 {len(results[0].boxes) if results else 0} 个物体")
