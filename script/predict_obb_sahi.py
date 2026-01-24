from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import os
import numpy as np

# 1. 配置参数
model_path = "/home/aistudio/YOLO/models/train/booth_obb_v1/weights/best.pt"  # 修改为OBB模型路径
source_image_path = "/home/aistudio/YOLO/images/第十一届世界猪业博览会.jpeg"
output_path = "/home/aistudio/YOLO/output_results/obb_result.jpg"

# 2. 加载模型 (使用 SAHI 的封装器，明确指定任务类型)
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.25,  # OBB任务可能调整置信度阈值
    device="cuda:0", 
    # 关键：指定模型任务类型为'obb'
    model_config_kwargs={'task': 'obb'}
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

# 4. 绘制结果：准备画布
img_original = cv2.imread(source_image_path)
img_obb = img_original.copy()  # 用于绘制旋转框

object_prediction_list = result.object_prediction_list

# 5. 解析OBB结果并绘制
for i, prediction in enumerate(object_prediction_list):
    # OBB任务的关键：获取旋转框信息
    # 对于OBB任务，Ultralytics返回的bbox可能是旋转框的四个点
    # 但SAHI可能将其转换为水平框，我们需要检查实际返回的数据结构
    
    # 方法1：检查是否有obb属性或特定字段
    # 首先尝试获取旋转框的四个点
    try:
        # 检查SAHI返回的bbox类型
        bbox = prediction.bbox
        
        # 对于OBB任务，我们需要获取旋转框的四个顶点
        # 这取决于SAHI如何解析Ultralytics OBB模型的输出
        
        # 尝试从预测结果中提取旋转框信息
        # 如果SAHI支持OBB，可能会有额外的属性
        
        # 方法A：如果有obb属性，直接使用
        if hasattr(prediction, 'obb') and prediction.obb is not None:
            # OBB格式：通常是四个点 (x1, y1, x2, y2, x3, y3, x4, y4)
            obb_points = prediction.obb
            points = np.array(obb_points).reshape(4, 2).astype(int)
            
            # 绘制旋转框的四个边
            for j in range(4):
                pt1 = tuple(points[j])
                pt2 = tuple(points[(j + 1) % 4])
                cv2.line(img_obb, pt1, pt2, (0, 0, 255), 2)  # 红色边
            
            # 绘制顶点
            for pt in points:
                cv2.circle(img_obb, tuple(pt), 5, (0, 255, 0), -1)  # 绿色顶点
            
            # 在框中心添加编号
            center = np.mean(points, axis=0).astype(int)
            cv2.putText(img_obb, f"{i}", tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            print(f"物体 {i}: 使用OBB点 {points.tolist()}")
            
        # 方法B：如果没有obb属性，尝试从bbox中提取
        else:
            # 检查bbox是否有额外信息（如旋转框的表示）
            # 如果SAHI不支持OBB，bbox将是水平框，我们就绘制水平框
            x1, y1 = int(bbox.minx), int(bbox.miny)
            x2, y2 = int(bbox.maxx), int(bbox.maxy)
            
            # 绘制水平框
            cv2.rectangle(img_obb, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 添加编号
            cv2.putText(img_obb, f"{i}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            print(f"物体 {i}: 使用水平框 ({x1}, {y1}, {x2}, {y2})")
            
    except Exception as e:
        print(f"处理物体 {i} 时出错: {e}")
        continue
    
    # 进度提示
    if (i + 1) % 50 == 0:
        print(f"已处理 {i + 1} / {len(object_prediction_list)} 个物体...")

# 6. 保存结果
cv2.imwrite(output_path, img_obb)
print("-" * 30)
print(f"OBB检测结果已保存: {output_path}")

# 7. 统计信息
print(f"\n检测统计:")
print(f"  总检测数: {len(object_prediction_list)}")
print(f"  图片尺寸: {img_original.shape[1]}x{img_original.shape[0]}")
print(f"  模型路径: {model_path}")

# 8. 可选：显示关键数据（用于调试）
print("\n前5个检测物体的详细信息:")
for i, prediction in enumerate(object_prediction_list[:5]):
    bbox = prediction.bbox
    score = prediction.score.value
    print(f"  物体 {i}: 置信度={score:.3f}, 位置=({bbox.minx:.1f}, {bbox.miny:.1f}, {bbox.maxx:.1f}, {bbox.maxy:.1f})")