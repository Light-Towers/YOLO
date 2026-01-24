from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import os
import numpy as np

# 1. 配置参数
model_path = "/home/aistudio/YOLO/models/train/booth_obb_v1/weights/best.pt"  # 修改为OBB模型路径
# source_image_path = "/home/aistudio/YOLO/images/第十一届世界猪业博览会.jpeg"
# source_image_path = "/home/aistudio/YOLO/images/2024年展位图_压缩.jpg"
source_image_path = "/home/aistudio/YOLO/images/2024年展位图.jpg"

# 从模型路径中提取版本信息（如 v1, v2 等）
# 假设模型路径格式为：.../booth_obb_v1/... 或 .../booth_obb_v2/...
import re
match = re.search(r'booth_obb_(v\d+)', model_path)
version = match.group(1) if match else "unknown_version"

# 构建输出目录结构
base_output_dir = "/home/aistudio/YOLO/output_results"
obb_result_dir = os.path.join(base_output_dir, "obb_result")
version_output_dir = os.path.join(obb_result_dir, version)

# 创建目录（如果不存在）
os.makedirs(version_output_dir, exist_ok=True)

# 从输入图像文件名生成输出文件名
input_filename = os.path.basename(source_image_path)
input_name, input_ext = os.path.splitext(input_filename)
output_filename = f"obb_result_{input_name}{input_ext}"
output_path = os.path.join(version_output_dir, output_filename)

print(f"模型版本: {version}")
print(f"输出目录: {version_output_dir}")
print(f"输出文件: {output_path}")

# 2. 加载模型 (使用 SAHI 的封装器，明确指定任务类型)
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.7,  # OBB任务可能调整置信度阈值
    device="cuda:0", 
)

print("正在进行切片推理，请稍候...")

# 3. 执行切片推理 
result = get_sliced_prediction(
    source_image_path,
    detection_model,
    slice_height=640,           # 切片高度
    slice_width=640,            # 切片宽度
    overlap_height_ratio=0.25,  # 高度重叠率
    overlap_width_ratio=0.25,   # 宽度重叠率
    postprocess_type="NMS",     # 合并算法
    postprocess_match_metric="IOS",  # 合并度量
    postprocess_match_threshold=0.3,  # OBB任务可能需要调整这个值
)

print(f"检测完成！共检测到 {len(result.object_prediction_list)} 个物体。")
# 4. 绘制结果：准备两张画布 (使用 OpenCV 手动绘制，模仿你之前的逻辑)
img_original = cv2.imread(source_image_path)
img_mask = img_original.copy()           # 用于绘制原始掩码轮廓
img_mask_rectangle = img_original.copy() # 用于绘制最小外接矩形

# 4. 绘制结果：准备画布
img_original = cv2.imread(source_image_path)
img_obb = img_original.copy()  # 用于绘制旋转框

object_prediction_list = result.object_prediction_list

# 使用 enumerate 解构出 i 和 j
for i, prediction in enumerate(object_prediction_list):
    # # 获取边界框坐标 [x_min, y_min, x_max, y_max]
    # bbox = prediction.bbox
    # x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
    # # 获取类别和置信度 (可选)
    # score = prediction.score.value
    # # 绘制红色框 (BGR: 0, 0, 255), 线宽 3
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)



    # 检查是否有分割掩码
    if prediction.mask is not None:
        # 1. 提取原始掩码
        # SAHI 的 mask.bool_mask 是一个与原图切片大小对应的布尔矩阵
        # 我们将其转为 uint8 格式 (0 或 255)
        mask_data = prediction.mask.bool_mask.astype(np.uint8) * 255
        
        # 使用 OpenCV 寻找掩码的轮廓
        contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        # --- 任务 1: 绘制原始掩码到 img_mask ---
        # 直接绘制所有原始轮廓
        # 参数说明：img, 轮廓列表, -1表示绘制所有轮廓, 颜色(红色), 线宽
        cv2.drawContours(img_mask, contours, -1, (0, 0, 255), 2)
        
        # --- 关键改动：只选取面积最大的轮廓 ---
        # 使用 max 函数配合 cv2.contourArea 快速找到最大项
        main_cnt = max(contours, key=cv2.contourArea)
        main_area = cv2.contourArea(main_cnt)
        
        # 3. --- 任务 2: 提取并绘制矩形到 img_mask_rectangle ---
        # 即使只剩一个轮廓，也可以保留面积过滤（比如过滤掉太小的误检物体）
        if main_area >= 100:
            # 提取最小外接矩形
            rect = cv2.minAreaRect(main_cnt) 
            box = cv2.boxPoints(rect).astype(int) 
            
            # 绘制规整矩形
            cv2.drawContours(img_mask_rectangle, [box], 0, (0, 0, 255), 2)
            
            # 在图上绘制唯一编号
            # 使用矩形中心点 (rect[0]) 标注，比顶点更不容易重叠
            text_x, text_y = int(rect[0][0]), int(rect[0][1])
            cv2.putText(img_mask_rectangle, f"{i}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            print(f"物体 [{i}] 的最大轮廓面积仅为 {main_area:.2f}，已被作为噪点过滤")
        
        
        
        # # --- 任务 2: 提取并绘制矩形到 img_mask_rectangle ---
        # for j, cnt in enumerate(contours):
        #     area = cv2.contourArea(cnt)     # 计算轮廓面积 (像素)
            
        #     if area < 400:
        #         # 打印物体索引 i 和 轮廓索引 j
        #         print(f"已过滤噪点：物体 [{i}] 的第 ({j}) 个轮廓面积仅为 {area:.2f}")
        #         continue
                
        #     # 提取最小外接矩形
        #     rect = cv2.minAreaRect(cnt)     # rect 的结构: ((x, y), (w, h), angle)
        #     box = cv2.boxPoints(rect)       # 获取矩形的四个顶点坐标
        #     box = box.astype(int)           # 转换为整数像素坐标
            
        #     # 绘制到第二张图上
        #     cv2.drawContours(img_mask_rectangle, [box], 0, (0, 0, 255), 2)
            
        #     # 在图上绘制编号
        #     ## 计算写字的位置（取矩形四个顶点的左上角，并稍微偏移一点）
        #     ## 或者直接使用 rect[0] (中心点)
        #     text_x, text_y = box[0][0], box[0][1] - 10
            
        #     ## 参数：图片, 文本, 位置, 字体, 缩放, 颜色(蓝色), 厚度
        #     cv2.putText(img_mask_rectangle, f"{i}", (text_x, text_y), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


        ##  使用 epsilon 逼近优化
        # for cnt in contours:
        #     # === 关键步骤：多边形逼近 ===
        #     # epsilon 是精确度参数，值越大越平滑（边越少）
        #     epsilon = 0.005 * cv2.arcLength(cnt, True) 
        #     approx = cv2.approxPolyDP(cnt, epsilon, True)
        #     # 绘制平滑后的轮廓
        #     cv2.polylines(img, [approx], True, (0, 0, 255), 2)


    else:
        # 如果没有 mask，则退而求其次绘制 bbox
        bbox = prediction.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        cv2.rectangle(img_mask, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(img_mask_rectangle, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 进度提示：每处理 200 个物体打印一次，防止程序看起来像卡死
    if (i + 1) % 200 == 0:
        print(f"已处理 {i + 1} / {len(object_prediction_list)} 个物体...")


# 5. 自动生成输出路径并保存
base_name, ext = os.path.splitext(output_path)
mask_output_path = f"{base_name}_mask{ext}"
rect_output_path = f"{base_name}_mask_rectangle{ext}"

cv2.imwrite(mask_output_path, img_mask)
cv2.imwrite(rect_output_path, img_mask_rectangle)

print("-" * 30)
print(f"原始掩码图已保存: {mask_output_path}")
print(f"规整矩形图已保存: {rect_output_path}")