from ultralytics import YOLO
import cv2

# 使用训练好的模型分析图片, 并根据返回的坐标在原图上标注

# model_path="/workspace/models/yolo11/yolo11n.pt"
model_path="/workspace/models/train/booth_detection_v1/weights/best.pt"

# 1. 加载预训练模型
model = YOLO(model_path)
# 2. 显示模型信息（可选）
model.info()
# 3. 训练模型（可选，如果不需要训练可以跳过）
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
# 4. 推理（目标检测）
# results = model("split_332_332.jpg")#在线下载
image = "/workspace/image/长沙国际会展中心.jpg"
results = model.predict(
    source=image,
    # iou=0.7,
    # imgsz=(2434,1484), #输入要分析图片的原始尺寸
    imgsz=(13051,13807), #输入要分析图片的原始尺寸
    # imgsz=640,
    device=0
)
# results = model("path/to/you/image", save=True)#本地照片
# 5. 显示结果
print("----------------------")
results[0].show()
print("----------------------")


def draw_rectangles_on_image(image_path, output_path, boxes, color=(0, 255, 0), thickness=2):
    """
    在图片上绘制多个矩形框（所有框使用相同颜色）。

    :param image_path: 输入图像路径
    :param output_path: 输出图像保存路径
    :param boxes: 坐标列表，格式为 [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    :param color: BGR 颜色，例如 (0, 255, 0) 表示绿色
    :param thickness: 线条粗细
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    # 遍历所有坐标框并绘制
    for box in boxes:
        x1, y1, x2, y2 = box
        # 可选：确保坐标是整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # 保存结果
    cv2.imwrite(output_path, img)
    print(f"✅ 已在 {len(boxes)} 个位置绘制矩形，结果保存至: {output_path}")


# 4. 获取并打印检测到的目标的位置信息
if len(results) > 0:
    boxes_list = []
    # 获取检测到的所有目标的信息
    boxes = results[0].boxes
    xyxy_tensor = results[0].boxes.xyxy  # shape: (N, 4)
    boxes_list = xyxy_tensor.cpu().numpy().astype(int).tolist()

    for box in boxes:
        # 边界框的坐标 (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # 置信度
        confidence = box.conf[0].item()
        # 类别
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]

        print(f"检测到目标: {class_name}, 置信度: {confidence:.2f}")
        print(f"位置信息: 左上角 ({x1}, {y1}), 右下角 ({x2}, {y2})")
    print("总数: ", len(results[0].boxes))

    output_path = "output_with_boxes.jpg"
    # 绘制所有框（统一用红色）
    draw_rectangles_on_image(
        image_path=image,
        output_path=output_path,
        boxes=boxes_list,
        color=(0, 0, 255),  # 红色 (BGR)
        thickness=3
    )






