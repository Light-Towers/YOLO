from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import json
import torch
from log_config import get_project_logger
import argparse
from pathlib import Path

# 获取项目logger
logger = get_project_logger('predict_obb_sahi')

def start_predict(model_path, image_path, dataset_name=None, output_dir=None):
    # 确保必须提供model_path和image_path参数
    if model_path is None:
        raise ValueError("model_path must be provided")
    if image_path is None:
        raise ValueError("image_path must be provided")

    model_path = str(model_path)
    image_path = str(image_path)

    # 从模型路径中提取模型名称
    model_name = Path(model_path).parent.parent.name  # 获取上级目录名，通常是模型名称
    if model_name == "weights":
        # 如果上两级目录是 weights 目录，则取再上一级目录名
        model_name = Path(model_path).parent.parent.parent.name

    # 如果仍无法获取有意义的模型名，则使用文件名的一部分
    if model_name in ["weights", "models", "results"]:
        model_file_name = Path(model_path).stem
        # 尝试从模型文件名中提取有意义的模型名
        parts = model_file_name.split('_')
        if len(parts) > 1:
            model_name = '_'.join(parts[:-1])  # 去掉最后一部分（如best.pt中的best）
        else:
            model_name = model_file_name

    # 构建输出目录结构
    project_dir = Path(__file__).resolve().parent.parent
    
    # 如果指定了输出目录则使用，否则使用默认目录
    if output_dir is not None:
        base_output_dir = Path(output_dir)
    else:
        base_output_dir = project_dir / "output" / "results"
    
    # 如果提供了数据集名称，使用它，否则尝试从模型路径推断
    if dataset_name is None:
        # 从模型路径尝试推断数据集名称
        parent_parts = str(Path(model_path).parent)
        # 查找可能的数据集名称部分
        if "booth" in parent_parts:
            dataset_name = "booth"
        else:
            dataset_name = "default_dataset"
    else:
        dataset_name = str(dataset_name)

    # 创建模型和数据集的输出目录
    model_dataset_dir = base_output_dir / model_name / dataset_name

    # 创建目录（如果不存在）
    os.makedirs(model_dataset_dir, exist_ok=True)

    # 从输入图像文件名生成输出文件名
    input_filename = os.path.basename(image_path)
    input_name, input_ext = os.path.splitext(input_filename)
    output_filename = f"{input_name}_sahi"
    output_path = model_dataset_dir / output_filename

    logger.info(f"模型名称: {model_name}")
    logger.info(f"数据集名称: {dataset_name}")
    logger.info(f"输出目录: {model_dataset_dir}")
    logger.info(f"输出文件: {output_path}.png")

    # 2. 加载模型 (使用 SAHI 的封装器，明确指定任务类型)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.7,  # OBB任务可能调整置信度阈值
        device="cuda:0" if torch.cuda.is_available() else "cpu", 
    )

    logger.info("正在进行切片推理，请稍候...")

    # 3. 执行切片推理 
    result = get_sliced_prediction(
        image_path,
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
        export_dir=str(model_dataset_dir),
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
            "bbox": prediction.bbox.to_xyxy(),
            "poly": prediction.mask.segmentation[0]  #  OBB 任务，尝试获取多边形点 (Segmentation/OBB)，这将捕获物体的实际形状
        })

    ## 保存为 JSON
    json_path = model_dataset_dir / f"{output_filename}.json"
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    logger.info(f"SAHI prediction completed. Results saved to {model_dataset_dir}")


if __name__ == "__main__":
    # 定义项目根目录：假设 script 在 project_root/script 下
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description='Run SAHI prediction with given model and image')
    parser.add_argument('--model_path', type=str, help='Path to the trained model',
                        default=str(project_root / 'models' / 'train' / 'booth_obb_v13' / 'weights' / 'best.pt'),
                        required=False)
    parser.add_argument('--image_path', type=str, help='Path to the image to predict',
                        default=str(project_root / 'images' / '第十一届世界猪业博览会.jpeg'),
                        required=False)
    parser.add_argument('--dataset_name', type=str, help='Dataset name to organize outputs', default=None)
    parser.add_argument('--output_dir', type=str, help='Directory to save prediction results', default=None)
    args = parser.parse_args()
    
    start_predict(args.model_path, args.image_path, args.dataset_name, args.output_dir)