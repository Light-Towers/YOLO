"""
切分器 - 只保留完整的展位标注，避免形状被切割
处理原始图片和json文件, 生成数据集
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import cv2
import numpy as np
from shapely.geometry import Polygon, box
import shapely.affinity as affinity
from pypinyin import lazy_pinyin
from log_config import get_project_logger

logger = get_project_logger('dataset_tiler')

class Tiler:
    """
    YOLO数据集切分器

    关键修复:
    1. 只保留完整在切片内的展位标注（不切割多边形）
    2. 增大overlap确保每个展位至少在一个切片中是完整的
    3. 可选：对于部分在切片内的展位，使用其完整边界框
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_path = Path(config["image_path"])
        self.json_path = Path(config["json_path"])
        # 对输出目录名进行中文转拼音处理
        original_output_dir = config["output_dir"]
        self.output_dir_name = self._convert_chinese_to_pinyin(Path(original_output_dir).name)
        self.output_dir = Path(original_output_dir).parent / self.output_dir_name
        self.tile_size = config.get("tile_size", 640)
        self.overlap = config.get("overlap", 200)  # 增大默认overlap
        self.split_ratio = config.get("split_ratio", 0.8)
        self.min_val_tiles = config.get("min_val_tiles", 2)
        self.class_names = config.get("class_names", ["booth"])
        self.dataset_name = self._convert_chinese_to_pinyin(config.get("dataset_name", "fixed_dataset"))
        
        # 新增配置：最小保留比例（展位面积在切片内的比例）
        self.min_area_ratio = config.get("min_area_ratio", 0.9)  # 90%以上才保留
        # 新增配置：是否只保留完整的4点多边形
        self.keep_only_complete = config.get("keep_only_complete", True)
        # 新增配置：是否保存JSON格式标注
        self.save_json = config.get("save_json", False)

        self._create_output_structure()
        
        self.img = cv2.imread(str(self.image_path))
        if self.img is None:
            raise ValueError(f"无法读取图像: {self.image_path}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.labelme_data = json.load(f)

        logger.info(f"🖼️  原图尺寸: {self.img.shape[1]}x{self.img.shape[0]}")
        logger.info(f"🏷️  标注对象数量: {len(self.labelme_data['shapes'])}")
        logger.info(f"📊 切片参数: size={self.tile_size}, overlap={self.overlap}")
        logger.info(f"⚙️  只保留完整标注: {self.keep_only_complete}")
        logger.info(f"⚙️  最小面积比例: {self.min_area_ratio:.0%}")
        logger.info(f"📁 输出目录: {self.output_dir}")

    def _create_output_structure(self):
        """创建输出目录结构"""
        for split in ['train', 'val']:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            if self.save_json:
                (self.output_dir / "json_annotations" / split).mkdir(parents=True, exist_ok=True)

        yaml_content = self._generate_yaml_content()
        (self.output_dir / "dataset.yaml").write_text(yaml_content, encoding='utf-8')
        logger.info(f"✅ 已创建数据集结构: {self.output_dir}")

    def _generate_yaml_content(self) -> str:
        path_str = str(self.output_dir.absolute())
        names_block = "names:\n"
        for i, name in enumerate(self.class_names):
            names_block += f"  {i}: {name}\n"

        return f"""# {self.dataset_name} - 修复版YOLO数据集配置
path: {path_str}
train: images/train
val: images/val

# 类别
{names_block}
"""

    def _get_all_tiles(self) -> List[Tuple[int, int, int, int, int]]:
        """获取所有切片位置"""
        h, w = self.img.shape[:2]
        tiles = []
        tile_id = 0
        step = self.tile_size - self.overlap

        y = 0
        while y < h:
            x = 0
            while x < w:
                x_end = min(x + self.tile_size, w)
                y_end = min(y + self.tile_size, h)
                tiles.append((tile_id, x, y, x_end, y_end))
                tile_id += 1
                x += step
            y += step

        return tiles

    def _assign_splits(self, tiles: list) -> Dict[str, list]:
        """分配训练/验证集"""
        total = len(tiles)
        val_count = max(self.min_val_tiles, int(total * (1 - self.split_ratio)))
        
        if total <= 2:
            val_count = 1
        
        train_tiles = tiles[:-val_count] if val_count < total else tiles[:1]
        val_tiles = tiles[-val_count:] if val_count > 0 else [tiles[-1]]

        logger.info(f"📊 数据集划分: 训练集 {len(train_tiles)}, 验证集 {len(val_tiles)}")
        return {'train': train_tiles, 'val': val_tiles}

    def _is_polygon_complete_in_tile(self, poly: Polygon, tile_box: box) -> bool:
        """检查多边形是否完整在切片内"""
        if not poly.intersects(tile_box):
            return False
        
        # 计算交集面积比例
        intersection = poly.intersection(tile_box)
        area_ratio = intersection.area / poly.area if poly.area > 0 else 0
        
        return area_ratio >= self.min_area_ratio

    # 添加中文转拼音方法
    def _convert_chinese_to_pinyin(self, text):
        """将中文转换为拼音"""
        if not text:
            return text

        try:
            pinyin_list = lazy_pinyin(text)
            result = ''.join(pinyin_list).lower()
            logger.info(f"🔤 '{text}' -> '{result}'")
            return result
        except:
            return text

    def _convert_annotation_fixed(self, shape: dict, x_offset: int, y_offset: int,
                                   tile_w: int, tile_h: int) -> Union[dict, None]:
        """
        修复版标注转换

        关键改变：不再切割多边形，只保留完整的展位
        """
        points = shape["points"]
        poly = Polygon(points)

        # 检查多边形有效性
        if not poly.is_valid:
            return None

        # 创建切片边界框
        tile_box = box(x_offset, y_offset, x_offset + tile_w, y_offset + tile_h)

        # 检查多边形是否完整在切片内
        if not self._is_polygon_complete_in_tile(poly, tile_box):
            return None

        # 获取原始shape_type
        original_shape_type = shape.get("shape_type", "polygon")

        # 根据配置决定处理方式
        if self.keep_only_complete:
            result = self._process_complete_polygon(points, x_offset, y_offset, tile_w, tile_h)
        else:
            result = self._process_intersected_polygon(poly, tile_box, x_offset, y_offset, tile_w, tile_h)

        # 保留原始shape_type
        if result:
            result["shape_type"] = original_shape_type

        return result
    
    def _process_complete_polygon(self, points: List[List[float]], x_offset: int, y_offset: int, 
                                  tile_w: int, tile_h: int) -> Union[dict, None]:
        """处理完整多边形"""
        # 直接使用原始多边形的点，不进行切割。 只有当多边形几乎完全在切片内时才使用原始点
        local_points = []
        for px, py in points:
            local_x = (px - x_offset) / tile_w
            local_y = (py - y_offset) / tile_h
            # 裁剪到[0, 1]范围
            local_x = max(0.0, min(1.0, local_x))
            local_y = max(0.0, min(1.0, local_y))
            local_points.append((local_x, local_y))
        
        # 验证点数（展位应该是4点四边形）
        if len(local_points) != 4:
            logger.warning(f"    ⚠️ 跳过非四边形标注 (点数: {len(local_points)})")
            return None
            
        return {
            "class_id": 0,
            "points": local_points,
            "original_points": len(points)
        }
    
    def _process_intersected_polygon(self, poly: Polygon, tile_box: box, 
                                     x_offset: int, y_offset: int, 
                                     tile_w: int, tile_h: int) -> Union[dict, None]:
        """处理相交多边形"""
        # 允许使用交集（但会改变形状）
        intersection = poly.intersection(tile_box)
        if intersection.is_empty or intersection.area < 100:
            return None
            
        local_poly = affinity.translate(intersection, xoff=-x_offset, yoff=-y_offset)
        
        try:
            coords = list(local_poly.exterior.coords)[:-1]
        except:
            return None
            
        normalized_points = [(px / tile_w, py / tile_h) for px, py in coords]
        
        return {
            "class_id": 0,
            "points": normalized_points,
            "original_points": len(list(poly.exterior.coords)[:-1])  # 原始点数
        }

    def process(self) -> dict:
        """执行切分"""
        all_tiles = self._get_all_tiles()
        logger.info(f"🔍 总计 {len(all_tiles)} 个切片位置")

        splits = self._assign_splits(all_tiles)
        results = {'train': [], 'val': []}
        
        # 统计信息
        stats = {
            'total_annotations': 0,
            'skipped_incomplete': 0,
            'kept_complete': 0
        }

        for split_name, tiles in splits.items():
            for tile_id, x, y, x_end, y_end in tiles:
                # 处理单个切片
                tile_result = self._process_tile(split_name, tile_id, x, y, x_end, y_end, stats)
                results[split_name].append(tile_result)

        # 打印统计
        self._print_statistics(all_tiles, results, stats)

        return {
            'output_dir': str(self.output_dir),
            'yaml_path': str(self.output_dir / "dataset.yaml"),
            'train_tiles': len(results['train']),
            'val_tiles': len(results['val']),
            'kept_complete': stats['kept_complete'],
            'skipped_incomplete': stats['skipped_incomplete']
        }
    
    def _process_tile(self, split_name: str, tile_id: int, x: int, y: int, x_end: int, y_end: int, stats: dict) -> dict:
        """处理单个切片"""
        tile_w, tile_h = x_end - x, y_end - y
        tile_img = self.img[y:y_end, x:x_end]
        
        # 对原始图片文件名进行中文转拼音处理
        original_stem = self.image_path.stem
        converted_stem = self._convert_chinese_to_pinyin(original_stem)
        tile_name = f"{converted_stem}_tile_{tile_id:04d}.png"

        # 保存图像
        img_path = self.output_dir / "images" / split_name / tile_name
        cv2.imwrite(str(img_path), tile_img)

        # 处理标注
        annotations = []
        for shape in self.labelme_data["shapes"]:
            # 只处理多边形和旋转框
            if shape["shape_type"] != "polygon" and shape["shape_type"] != "rotation":
                continue

            stats['total_annotations'] += 1
            ann = self._convert_annotation_fixed(shape, x, y, tile_w, tile_h)
            
            if ann:
                annotations.append(ann)
                stats['kept_complete'] += 1
            else:
                stats['skipped_incomplete'] += 1

        # 保存标注
        lbl_path = self.output_dir / "labels" / split_name / tile_name.replace(".png", ".txt")
        with open(lbl_path, 'w') as f:
            for ann in annotations:
                points_str = " ".join([f"{px:.6f} {py:.6f}" for px, py in ann["points"]])
                f.write(f"0 {points_str}\n")

        # 保存JSON格式标注（用于标注工具检查）
        if self.save_json:
            self._save_json_annotation(split_name, tile_name, tile_w, tile_h, annotations)

        status = "✅" if annotations else "🟡"
        logger.info(f"{status} {split_name}: {tile_name} - {len(annotations)} 个完整展位")

        return {
            'name': tile_name,
            'annotations': len(annotations),
            'position': (x, y, x_end, y_end)
        }

    def _save_json_annotation(self, split_name: str, tile_name: str, tile_w: int, tile_h: int, annotations: List[dict]):
        """保存JSON格式标注（用于标注工具检查）"""
        # 转换为像素坐标
        shapes = []
        for ann in annotations:
            points = [[px * tile_w, py * tile_h] for px, py in ann["points"]]
            shape_type = ann.get("shape_type", "polygon")  # 从annotation中获取shape_type
            shapes.append({
                "label": self.class_names[ann["class_id"]],
                "points": points,
                "group_id": None,
                "shape_type": shape_type,
                "flags": {}
            })

        # 生成JSON数据
        json_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": tile_name,
            "imageData": None,
            "imageHeight": tile_h,
            "imageWidth": tile_w
        }

        # 保存JSON文件
        json_path = self.output_dir / "json_annotations" / split_name / tile_name.replace(".png", ".json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    def _print_statistics(self, all_tiles: list, results: dict, stats: dict):
        """打印统计信息"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 切分统计报告")
        logger.info("=" * 60)
        logger.info(f"总切片数: {len(all_tiles)}")
        logger.info(f"训练集标注: {sum(t['annotations'] for t in results['train'])}")
        logger.info(f"验证集标注: {sum(t['annotations'] for t in results['val'])}")
        logger.info(f"保留的完整展位: {stats['kept_complete']}")
        logger.info(f"跳过的不完整展位: {stats['skipped_incomplete']}")
        logger.info(f"保留率: {stats['kept_complete'] / max(stats['total_annotations'], 1):.1%}")
        logger.info("=" * 60)


def find_matching_image(base_name: str, image_dir: Path) -> Path:
    """根据JSON文件名查找匹配的图片文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # 首先尝试精确匹配
    for ext in image_extensions:
        image_path = image_dir / f"{base_name}{ext}"
        if image_path.exists():
            return image_path
    
    # 如果没有找到精确匹配，尝试在目录中查找相似名称的文件
    for img_file in image_dir.iterdir():
        if img_file.is_file() and img_file.stem == base_name and img_file.suffix.lower() in image_extensions:
            return img_file

    raise FileNotFoundError(f"找不到与 {base_name} 对应的图片文件")


def process_json_file(json_path: Path, image_dir: Path = Path("images")):
    """处理单个JSON文件"""
    # 获取JSON文件的基础名称（不含扩展名）
    json_stem = json_path.stem
    image_path = find_matching_image(json_stem, image_dir)

    config = {
        "image_path": str(image_path),
        "json_path": str(json_path),
        "output_dir": f"datasets/{json_stem}",  # 根据JSON文件名生成输出目录名
        
        # 切片参数 - 关键修改
        "tile_size": 640,
        "overlap": 200,  # 增大overlap，确保更多展位在某个切片中是完整的        
        # 数据集参数
        "split_ratio": 0.8,
        "min_val_tiles": 3,
        "class_names": ["booth"],  # 使用更有意义的类名
        "dataset_name": json_stem,
        "min_area_ratio": 0.85,  # 展位85%以上在切片内才保留
        "keep_only_complete": True,  # 只保留完整的4点四边形
        "save_json": False,  # 是否保存JSON格式标注（默认关闭）
    }

    logger.info(f"🔧 使用配置: {json_stem}")
    logger.info(f"📄 JSON文件: {json_path.name}")
    logger.info(f"🖼️  匹配图片: {image_path.name}")

    # 创建切分器并执行
    tiler = Tiler(config)
    result = tiler.process()

    logger.info(f"\n✅ 数据集已生成: {result['output_dir']}")
    logger.info(f"📄 YAML配置: {result['yaml_path']}")

def main(input_source: str = r"labelme_annotations/11-ZhuYe.json"):
    """主函数 - 用于切分"""
    image_dir = Path("images")
    input_path = Path(input_source)

    # 处理单个JSON文件
    if input_path.is_file() and input_path.suffix.lower() == '.json':
        logger.info(f"📁 处理单个JSON文件: {input_path}")
        process_json_file(input_path, image_dir)
    # 处理文件夹
    elif input_path.is_dir():
        logger.info(f"📂 处理文件夹: {input_path}")
        json_files = list(input_path.glob('*.json'))
        if not json_files:
            logger.warning(f"⚠️  在 {input_path} 中未找到JSON文件")
        else:
            logger.info(f"🔍 找到 {len(json_files)} 个JSON文件")
            for json_file in json_files:
                logger.info(f"  📄 {json_file.name}")
                process_json_file(json_file, image_dir)
    # 处理多个JSON文件列表（逗号分隔）
    elif ',' in input_source:
        logger.info("📚 处理多个JSON文件列表")
        for path_str in input_source.split(','):
            json_file = Path(path_str.strip())
            if json_file.is_file():
                logger.info(f"  📄 {json_file.name}")
                process_json_file(json_file, image_dir)
            else:
                logger.error(f"  ❌ 文件不存在: {json_file}")
    else:
        logger.error(f"❌ 输入路径无效: {input_path}")
        logger.error("💡 请提供有效的JSON文件路径、文件夹路径或逗号分隔的多个文件路径")


if __name__ == "__main__":
    # 可以在这里指定input_source参数，如果不指定则使用默认值
    input_source = r"labelme_annotations/测试切图_222.json"
    main(input_source)