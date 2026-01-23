from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import numpy as np

# 1. 配置参数
model_path = "/workspace/models/train/booth_seg_v19/weights/best.pt"
# source_image_path = "/workspace/image/2025年畜牧-展位分布图-1105-01.png"
# source_image_path = "/workspace/image/第十一届世界猪业博览会.jpeg"
# source_image_path = "/workspace/image/2024年展位图.jpg"
# source_image_path = "/workspace/image/2024年展位图-test.png"
source_image_path = "/workspace/image/2024年展位图_压缩.jpg"
output_path = "/workspace/output_results/sahi_result_3.jpg"

# 2. 加载模型 (使用 SAHI 的封装器)
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.7,
    device="cuda:0", 
)

print("正在进行切片推理，请稍候...")

# 3. 执行切片推理 
# 这会自动把大图切成 640x640 的小块进行预测，然后合并结果
result = get_sliced_prediction(
    source_image_path,
    detection_model,
    slice_height=640,        # 切片高度
    slice_width=640,         # 切片宽度
    overlap_height_ratio=0.25, # 高度重叠率 (防止物体被切成两半)
    overlap_width_ratio=0.25,   # 宽度重叠率
    # === 加强合并逻辑 ===
    postprocess_type="NMS",   # 显式指定使用 NMS 合并
    # postprocess_match_metric="iou", # 或者尝试 "ios" (Intersection over Smaller area)
    postprocess_match_threshold=0.45, # 降低这个值会合并更多重叠框（如果框重叠30%就合并）
)

print(f"检测完成！共检测到 {len(result.object_prediction_list)} 个物体。")

# 4. 绘制结果 (使用 OpenCV 手动绘制，模仿你之前的逻辑)
img = cv2.imread(source_image_path)

object_prediction_list = result.object_prediction_list

for prediction in object_prediction_list:
    # # 获取边界框坐标 [x_min, y_min, x_max, y_max]
    # bbox = prediction.bbox
    # x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
    # # 获取类别和置信度 (可选)
    # score = prediction.score.value
    # # 绘制红色框 (BGR: 0, 0, 255), 线宽 3
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)



    # 检查是否有分割掩码
    if prediction.mask is not None:
        # SAHI 的 mask.bool_mask 是一个与原图切片大小对应的布尔矩阵
        # 我们将其转为 uint8 格式 (0 或 255)
        mask_data = prediction.mask.bool_mask.astype(np.uint8) * 255
        
        # 使用 OpenCV 寻找掩码的轮廓
        contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # # 绘制轮廓 (红色，线宽 2)
        # cv2.drawContours(img, contours, -1, (0, 0, 255), 2)


        for cnt in contours:
            # === 关键步骤：多边形逼近 ===
            # epsilon 是精确度参数，值越大越平滑（边越少）
            epsilon = 0.005 * cv2.arcLength(cnt, True) 
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # 绘制平滑后的轮廓
            cv2.polylines(img, [approx], True, (0, 0, 255), 2)


    else:
        # 如果没有 mask，则退而求其次绘制 bbox
        bbox = prediction.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)



# 5. 保存图片
cv2.imwrite(output_path, img)
print(f"结果已保存至: {output_path}")