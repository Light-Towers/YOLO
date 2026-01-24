from ultralytics import YOLO
import cv2
import os
import json
import numpy as np
from pathlib import Path
import pickle

class BoothSegmentationPredictor:
    def __init__(self, model_path=None, output_dir=None):
        """初始化预测器
        
        Args:
            model_path: 模型权重文件路径（相对或绝对路径）
            output_dir: 输出目录
        """
        # 获取脚本所在目录
        self.script_dir = Path(__file__).parent.absolute()
        # 项目根目录（假设脚本在script/目录下）
        self.project_root = self.script_dir.parent
        
        print(f"脚本目录: {self.script_dir}")
        print(f"项目根目录: {self.project_root}")
        
        # 设置默认模型路径（基于项目结构）
        if model_path is None:
            # 默认使用最近训练的模型
            model_path = self.project_root / "models" / "train" / "booth_seg_v17" / "weights" / "best.pt"
        else:
            # 如果提供的是相对路径，转换为绝对路径
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = self.project_root / model_path
        
        self.model_path = str(model_path)
        
        # 设置默认输出目录
        if output_dir is None:
            output_dir = self.project_root / "output_results"
        else:
            output_dir = Path(output_dir)
            if not output_dir.is_absolute():
                output_dir = self.project_root / output_dir
        
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        print(f"正在加载模型: {self.model_path}")
        if not Path(self.model_path).exists():
            print(f"错误: 模型文件不存在: {self.model_path}")
            print("可用模型文件:")
            models_dir = self.project_root / "models" / "train"
            if models_dir.exists():
                for model_folder in models_dir.iterdir():
                    if model_folder.is_dir():
                        weights_dir = model_folder / "weights"
                        if weights_dir.exists():
                            for weight_file in weights_dir.glob("*.pt"):
                                print(f"  - {weight_file.relative_to(self.project_root)}")
            return
        
        self.model = YOLO(self.model_path)
        print("模型加载完成")
    
    def predict(self, source_image=None, conf=0.7, iou=0.4, imgsz=None):
        """执行预测并保存结果到文件
        
        Args:
            source_image: 输入图像路径（相对或绝对路径）
            conf: 置信度阈值
            iou: IoU阈值
            imgsz: 图像尺寸 (宽度, 高度)
            
        Returns:
            预测结果保存路径
        """
        # 设置默认图像路径
        if source_image is None:
            # 默认使用测试图像
            source_image = self.project_root / "images" / "2024年展位图_压缩.jpg"
        else:
            # 如果提供的是相对路径，转换为绝对路径
            source_image = Path(source_image)
            if not source_image.is_absolute():
                source_image = self.project_root / source_image
        
        self.source_image = str(source_image)
        self.image_name = Path(source_image).stem
        
        # 检查图像文件是否存在
        if not Path(self.source_image).exists():
            print(f"错误: 图像文件不存在: {self.source_image}")
            print("可用图像文件:")
            images_dir = self.project_root / "images"
            if images_dir.exists():
                for img_file in images_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        print(f"  - {img_file.name}")
            return None
        
        # 如果未指定imgsz，则自动获取图像尺寸
        if imgsz is None:
            img = cv2.imread(self.source_image)
            if img is not None:
                # 获取原始图像尺寸
                h, w = img.shape[:2]
                imgsz = (w, h)  # (宽度, 高度)
                print(f"图像原始尺寸: {w} x {h}")
                
                # 自动调整尺寸，保持长宽比
                max_size = 1280  # 最大尺寸限制，避免显存溢出
                if max(w, h) > max_size:
                    scale = max_size / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    imgsz = (new_w, new_h)
                    print(f"自动调整尺寸至: {new_w} x {new_h}")
            else:
                imgsz = (640, 640)
                print(f"使用默认尺寸: {imgsz}")
        
        print(f"正在对图像进行预测: {Path(self.source_image).name}")
        print(f"使用图像尺寸: {imgsz}")
        
        # 执行预测
        try:
            results = self.model.predict(
                source=self.source_image,
                save=False,  # 不自动保存，避免显存问题
                conf=conf,
                iou=iou,
                device=0,
                project=str(self.output_dir),
                name="predictions",
                exist_ok=True,
                imgsz=imgsz,
                verbose=False  # 减少控制台输出
            )
        except Exception as e:
            print(f"预测过程中出错: {e}")
            print("尝试调整图像尺寸...")
            # 尝试使用较小尺寸
            results = self.model.predict(
                source=self.source_image,
                save=False,
                conf=conf,
                iou=iou,
                device=0,
                project=str(self.output_dir),
                name="predictions",
                exist_ok=True,
                imgsz=640,  # 使用固定尺寸
                verbose=False
            )
        
        # 提取并保存预测结果
        result_data = self._extract_results(results[0])
        
        # 保存结果到文件
        result_file = self._save_results(result_data)
        
        return result_file
    
    def _extract_results(self, result):
        """从预测结果中提取有用信息
        
        Args:
            result: 单个预测结果
            
        Returns:
            包含预测信息的字典
        """
        result_data = {
            "image_path": self.source_image,
            "image_name": self.image_name,
            "num_detections": 0,
            "boxes": [],
            "masks": [],
            "confidences": [],
            "classes": []
        }
        
        # 提取边界框信息
        if result.boxes is not None and len(result.boxes) > 0:
            result_data["num_detections"] = len(result.boxes)
            
            for i, box in enumerate(result.boxes):
                # 边界框坐标
                box_coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                result_data["boxes"].append(box_coords)
                result_data["confidences"].append(confidence)
                result_data["classes"].append(class_id)
        
        # 提取分割掩码信息
        if result.masks is not None and len(result.masks) > 0:
            for i, mask in enumerate(result.masks):
                # 获取多边形坐标
                segments = mask.xy
                mask_polygons = []
                
                for segment in segments:
                    # 转换为列表格式
                    polygon = segment.tolist()
                    mask_polygons.append(polygon)
                
                result_data["masks"].append(mask_polygons)
        
        print(f"检测到 {result_data['num_detections']} 个展位")
        return result_data
    
    def _save_results(self, result_data):
        """保存预测结果到文件
        
        Args:
            result_data: 预测结果数据
            
        Returns:
            保存的文件路径
        """
        # 创建结果目录
        results_dir = self.output_dir / "results_data"
        results_dir.mkdir(exist_ok=True)
        
        # 保存为JSON文件（人类可读）
        json_file = results_dir / f"{self.image_name}_results.json"
        with open(json_file, 'w') as f:
            # 将numpy数组转换为列表
            json_data = result_data.copy()
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # 保存为pickle文件（保留完整数据）
        pkl_file = results_dir / f"{self.image_name}_results.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(result_data, f)
        
        # 保存简化的文本摘要
        txt_file = results_dir / f"{self.image_name}_summary.txt"
        with open(txt_file, 'w') as f:
            f.write(f"图像: {result_data['image_name']}\n")
            f.write(f"检测数量: {result_data['num_detections']}\n")
            f.write(f"保存时间: {result_data.get('save_time', 'N/A')}\n")
            f.write("\n检测详情:\n")
            for i, (box, conf) in enumerate(zip(result_data['boxes'], result_data['confidences'])):
                f.write(f"检测框 {i+1}: 坐标 {box}, 置信度 {conf:.4f}\n")
        
        print(f"预测结果已保存到:")
        print(f"  JSON文件: {json_file.relative_to(self.project_root)}")
        print(f"  Pickle文件: {pkl_file.relative_to(self.project_root)}")
        print(f"  文本摘要: {txt_file.relative_to(self.project_root)}")
        
        return str(json_file)
    
    def draw_results(self, result_file=None, output_image_path=None, 
                    box_color=(0, 0, 255), box_thickness=2,
                    mask_color=(0, 255, 0), mask_thickness=1,
                    draw_boxes=True, draw_masks=True,
                    show_labels=False, label_color=(255, 255, 255),
                    image_name=None):
        """根据保存的结果文件绘制预测结果
        
        Args:
            result_file: 结果文件路径（JSON或pickle），如果为None则查找最新结果
            output_image_path: 输出图像路径
            box_color: 边界框颜色 (B, G, R)
            box_thickness: 边界框线宽
            mask_color: 掩码轮廓颜色
            mask_thickness: 掩码轮廓线宽
            draw_boxes: 是否绘制边界框
            draw_masks: 是否绘制掩码轮廓
            show_labels: 是否显示标签（类别和置信度）
            label_color: 标签颜色
            image_name: 图像名称（用于查找结果文件）
            
        Returns:
            绘制后的图像
        """
        # 如果没有指定结果文件，尝试自动查找
        if result_file is None:
            results_dir = self.output_dir / "results_data"
            if not results_dir.exists():
                print(f"错误: 结果目录不存在: {results_dir}")
                return None
            
            # 如果指定了图像名称，查找对应的结果文件
            if image_name:
                json_file = results_dir / f"{image_name}_results.json"
                if json_file.exists():
                    result_file = str(json_file)
                else:
                    print(f"未找到图像 {image_name} 的结果文件")
                    return None
            else:
                # 查找最新的结果文件
                json_files = list(results_dir.glob("*_results.json"))
                if not json_files:
                    print("未找到任何结果文件")
                    return None
                
                # 按修改时间排序，获取最新的文件
                latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
                result_file = str(latest_file)
                print(f"使用最新的结果文件: {latest_file.name}")
        
        # 加载预测结果
        result_file = Path(result_file)
        if not result_file.exists():
            # 尝试在项目目录中查找
            if not result_file.is_absolute():
                result_file = self.project_root / result_file
            
            if not result_file.exists():
                print(f"错误: 结果文件不存在: {result_file}")
                return None
        
        if result_file.suffix == '.json':
            with open(result_file, 'r') as f:
                result_data = json.load(f)
        elif result_file.suffix == '.pkl':
            with open(result_file, 'rb') as f:
                result_data = pickle.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {result_file.suffix}")
        
        # 读取原始图像
        image_path = result_data["image_path"]
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        print(f"正在绘制预测结果到图像: {Path(image_path).name}")
        
        # 绘制边界框
        if draw_boxes and result_data["boxes"]:
            for i, box in enumerate(result_data["boxes"]):
                x1, y1, x2, y2 = map(int, box)
                
                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)
                
                # 绘制标签（可选）
                if show_labels and i < len(result_data["confidences"]):
                    conf = result_data["confidences"][i]
                    class_id = result_data["classes"][i]
                    
                    label = f"Booth {class_id}: {conf:.2f}"
                    
                    # 计算标签位置
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                    
                    # 绘制标签背景
                    cv2.rectangle(img, (x1, label_y - label_size[1]), 
                                (x1 + label_size[0], label_y + 5), box_color, -1)
                    
                    # 绘制标签文字
                    cv2.putText(img, label, (x1, label_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
        
        # 绘制分割掩码轮廓
        if draw_masks and result_data["masks"]:
            for mask_polygons in result_data["masks"]:
                for polygon in mask_polygons:
                    if polygon:  # 确保多边形不为空
                        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], isClosed=True, 
                                    color=mask_color, thickness=mask_thickness)
        
        # 保存图像
        if output_image_path is None:
            style = "boxes" if draw_boxes and not draw_masks else "masks" if not draw_boxes and draw_masks else "both"
            output_image_path = self.output_dir / f"{result_data['image_name']}_{style}.jpg"
        else:
            output_image_path = Path(output_image_path)
            if not output_image_path.is_absolute():
                output_image_path = self.project_root / output_image_path
        
        cv2.imwrite(str(output_image_path), img)
        print(f"绘制完成，图像保存至: {output_image_path.relative_to(self.project_root)}")
        
        return img

    def list_available_images(self):
        """列出可用的测试图像"""
        images_dir = self.project_root / "images"
        if not images_dir.exists():
            print(f"图像目录不存在: {images_dir}")
            return []
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_dir.glob(ext))
        
        print("可用测试图像:")
        for i, img_file in enumerate(sorted(image_files)):
            print(f"  {i+1}. {img_file.name}")
        
        return image_files

# 实用函数
def load_results(result_file):
    """加载保存的预测结果"""
    result_file = Path(result_file)
    if result_file.suffix == '.json':
        with open(result_file, 'r') as f:
            return json.load(f)
    elif result_file.suffix == '.pkl':
        with open(result_file, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {result_file.suffix}")

def visualize_single_detection(result_file, detection_idx=0, output_dir=None):
    """可视化单个检测结果，用于调试"""
    result_data = load_results(result_file)
    img = cv2.imread(result_data["image_path"])
    
    if detection_idx < len(result_data["boxes"]):
        box = result_data["boxes"][detection_idx]
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制该检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 设置输出路径
        if output_dir is None:
            script_dir = Path(__file__).parent.absolute()
            output_dir = script_dir.parent / "output_results" / "debug"
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"detection_{detection_idx}_{Path(result_data['image_name']).stem}.jpg"
        cv2.imwrite(str(output_path), img)
        print(f"单个检测结果保存至: {output_path.relative_to(Path(__file__).parent.parent)}")
        
        # 打印详细信息
        print(f"检测框 {detection_idx}:")
        print(f"  坐标: [{x1}, {y1}, {x2}, {y2}]")
        print(f"  置信度: {result_data['confidences'][detection_idx]:.4f}")
        
        return img
    else:
        print(f"错误: 检测索引 {detection_idx} 超出范围 (总共 {len(result_data['boxes'])} 个检测)")
        return None

def analyze_results_statistics(result_file):
    """分析预测结果的统计信息"""
    result_data = load_results(result_file)
    
    print("=" * 50)
    print("预测结果统计信息")
    print("=" * 50)
    print(f"图像: {result_data['image_name']}")
    print(f"检测数量: {result_data['num_detections']}")
    
    if result_data['num_detections'] > 0:
        confidences = result_data['confidences']
        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"最高置信度: {np.max(confidences):.4f}")
        print(f"最低置信度: {np.min(confidences):.4f}")
        
        # 统计边界框大小
        boxes = np.array(result_data['boxes'])
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        print(f"平均边界框大小: {np.mean(widths):.1f} x {np.mean(heights):.1f}")
        print(f"平均面积: {np.mean(areas):.1f} 像素")
        print(f"最大边界框: {np.max(widths):.1f} x {np.max(heights):.1f}")
        print(f"最小边界框: {np.min(widths):.1f} x {np.min(heights):.1f}")
    
    return result_data

# 使用示例 - 添加命令行支持
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='展位分割预测工具')
    parser.add_argument('--model', type=str, default=None, 
                       help='模型路径 (相对或绝对路径)')
    parser.add_argument('--image', type=str, default=None,
                       help='测试图像路径 (相对或绝对路径)')
    parser.add_argument('--conf', type=float, default=0.7,
                       help='置信度阈值 (默认: 0.7)')
    parser.add_argument('--iou', type=float, default=0.4,
                       help='IoU阈值 (默认: 0.4)')
    parser.add_argument('--list-images', action='store_true',
                       help='列出可用测试图像')
    parser.add_argument('--draw-only', action='store_true',
                       help='仅绘制已保存的结果，不进行预测')
    parser.add_argument('--image-name', type=str,
                       help='指定图像名称用于绘制结果')
    
    args = parser.parse_args()
    
    # 1. 初始化预测器
    predictor = BoothSegmentationPredictor(model_path=args.model)
    
    # 如果只是列出图像
    if args.list_images:
        predictor.list_available_images()
        return
    
    # 如果只是绘制结果
    if args.draw_only:
        if args.image_name:
            predictor.draw_results(image_name=args.image_name)
        else:
            predictor.draw_results()
        return
    
    # 2. 执行预测并保存结果
    result_file = predictor.predict(
        source_image=args.image,
        conf=args.conf,
        iou=args.iou
    )
    
    if result_file:
        # 3. 绘制结果
        # 示例1: 绘制红色边界框
        predictor.draw_results(
            result_file,
            output_image_path="red_boxes.jpg",
            box_color=(0, 0, 255),
            box_thickness=3,
            draw_boxes=True,
            draw_masks=False,
            show_labels=True
        )
        
        # 示例2: 绘制绿色掩码轮廓
        predictor.draw_results(
            result_file,
            output_image_path="green_masks.jpg",
            mask_color=(0, 255, 0),
            mask_thickness=2,
            draw_boxes=False,
            draw_masks=True
        )
        
        # 示例3: 同时绘制边界框和掩码
        predictor.draw_results(
            result_file,
            output_image_path="both.jpg",
            box_color=(255, 0, 0),
            box_thickness=2,
            mask_color=(0, 165, 255),
            mask_thickness=1,
            draw_boxes=True,
            draw_masks=True,
            show_labels=True
        )
        
        # 分析结果统计
        analyze_results_statistics(result_file)
        
        print("\n🎉 预测完成！所有结果已保存到 output_results/ 目录")

if __name__ == "__main__":
    main()