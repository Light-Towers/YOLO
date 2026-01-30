from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import json
from log_config import get_project_logger

# 获取项目logger
logger = get_project_logger('predict_obb_sahi')

# 1. 配置参数
# model_path = "/home/aistudio/YOLO/models/train/booth_obb_v4/weights/best.pt"  # 修改为OBB模型路径
# model_path = "/home/aistudio/YOLO/models/train/booth_obb_mix_booth_obb_v1/weights/best.pt"  # 修改为OBB模型路径
# model_path = "/home/aistudio/YOLO/models/train/booth_obb_fixed_tiled_dataset_1024_v1/weights/best.pt"  # 修改为OBB模型路径
model_path = "/home/aistudio/YOLO/models/train/booth_obb_mix_booth_obb_v12/weights/best.pt"  # 修改为OBB模型路径
## YOLO26
# model_path = "/home/aistudio/YOLO/models/train/booth_obb_mix_booth_obb_yolo26_1024_v1/weights/best.pt"

# source_image_path = "/home/aistudio/YOLO/images/第十一届世界猪业博览会.jpeg"
# source_image_path = "/home/aistudio/YOLO/images/长沙国际会展中心.jpg"
# source_image_path = "/home/aistudio/YOLO/images/2020畜博会.png"
# source_image_path = "/home/aistudio/YOLO/images/企业微信截图_17695887706021.png"
# source_image_path = "/home/aistudio/YOLO/images/2025年畜牧-展位分布图-1105-01.png"
# source_image_path = "/home/aistudio/YOLO/images/2024年展位图_压缩.jpg"
source_image_path = "/home/aistudio/YOLO/images/2024年展位图.jpg"

# 从模型路径中提取版本信息（如 v1, v2 等）
import re
match = re.search(r'_(v\d+)', model_path)
version = match.group(1) if match else "unknown_version"

# 构建输出目录结构
base_output_dir = "/home/aistudio/YOLO/output_results"
# obb_result_dir = os.path.join(base_output_dir, "fixed_tiled_dataset_1024")
obb_result_dir = os.path.join(base_output_dir, "mix_booth_obb_result")
version_output_dir = os.path.join(obb_result_dir, f"{version}_src1024")
# version_output_dir = os.path.join(obb_result_dir, f"{version}_yolo26_src1024")

# 创建目录（如果不存在）
os.makedirs(version_output_dir, exist_ok=True)

# 从输入图像文件名生成输出文件名
input_filename = os.path.basename(source_image_path)
input_name, input_ext = os.path.splitext(input_filename)
output_filename = f"obb_result_{input_name}_{version}"
output_path = os.path.join(version_output_dir, output_filename)

logger.info(f"模型版本: {version}")
logger.info(f"输出目录: {version_output_dir}")
logger.info(f"输出文件: {output_path}.png")

# 2. 加载模型 (使用 SAHI 的封装器，明确指定任务类型)
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.7,  # OBB任务可能调整置信度阈值
    device="cuda:0", 
)

logger.info("正在进行切片推理，请稍候...")

# 3. 执行切片推理 
result = get_sliced_prediction(
    source_image_path,
    detection_model,
    slice_height=1024,           # 切片高度
    slice_width=1024,            # 切片宽度
    overlap_height_ratio=0.5,  # 高度重叠率
    overlap_width_ratio=0.5,   # 宽度重叠率
    postprocess_type="NMS",     # 合并算法
    postprocess_match_metric="IOS",  # 合并度量
    postprocess_match_threshold=0.5,  # OBB任务可能需要调整这个值
)

logger.info(f"检测完成！共检测到 {len(result.object_prediction_list)} 个物体。")

# 4. 导出结果
result.export_visuals(
    export_dir=version_output_dir,
    hide_labels=True,      # 隐藏类别名称
    hide_conf=True,        # 隐藏置信度数值
    rect_th=2,              # 调小线条粗细（默认通常为 2 或 3）
    file_name=f"{output_filename}",
)

## 直接访问 SAHI 的 object_prediction_list 获取检测信息
summary_data = []
for prediction in result.object_prediction_list:
    summary_data.append({
        "name": prediction.category.name,
        "class": prediction.category.id,
        "confidence": float(prediction.score.value),
        "poly": prediction.mask.segmentation[0]  #  OBB 任务，尝试获取多边形点 (Segmentation/OBB)，这将捕获物体的实际形状
    })

## 保存为 JSON
with open(f"{output_filename}.json", "w") as f:
    json.dump(summary_data, f, indent=4)