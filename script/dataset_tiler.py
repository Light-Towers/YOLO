"""
切分器 - 只保留完整的展位标注，避免形状被切割
处理原始图片和json文件, 生成数据集
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import shutil
import cv2
from shapely.geometry import Polygon, box
import shapely.affinity as affinity
from pypinyin import lazy_pinyin

# 导入工程化工具
from src.utils import (
    get_logger,
    safe_mkdir,
    read_json,
    write_json,
)
from src.utils.image_tile_utils import TileCalculator
from src.core import DATASET_CONSTANTS

logger = get_logger('dataset_tiler')

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
        self.tile_size = config.get("tile_size", DATASET_CONSTANTS.DEFAULT_TILE_SIZE)
        self.overlap = config.get("overlap", DATASET_CONSTANTS.DEFAULT_OVERLAP)
        self.split_ratio = config.get("split_ratio", DATASET_CONSTANTS.DEFAULT_TRAIN_RATIO)
        self.min_val_tiles = config.get("min_val_tiles", DATASET_CONSTANTS.DEFAULT_MIN_VAL_TILES)
        self.class_names = config.get("class_names", ["booth"])
        self.dataset_name = self._convert_chinese_to_pinyin(config.get("dataset_name", "fixed_dataset"))

        # 新增配置：最小保留比例（展位面积在切片内的比例）
        self.min_area_ratio = config.get("min_area_ratio", DATASET_CONSTANTS.DEFAULT_MIN_AREA_RATIO)
        # 新增配置：是否只保留完整的4点多边形
        self.keep_only_complete = config.get("keep_only_complete", True)
        # 新增配置：是否保存JSON格式标注
        self.save_json = config.get("save_json", False)

        self._create_output_structure()

        self.img = cv2.imread(str(self.image_path))
        if self.img is None:
            raise ValueError(f"无法读取图像: {self.image_path}")

        # 使用工具函数读取JSON
        self.labelme_data = read_json(self.json_path)

        logger.info(f"🖼️  原图尺寸: {self.img.shape[1]}x{self.img.shape[0]}")
        logger.info(f"🏷️  标注对象数量: {len(self.labelme_data['shapes'])}")
        logger.info(f"📊 切片参数: size={self.tile_size}, overlap={self.overlap}")
        logger.info(f"⚙️  只保留完整标注: {self.keep_only_complete}")
        logger.info(f"⚙️  最小面积比例: {self.min_area_ratio:.0%}")
        logger.info(f"📁 输出目录: {self.output_dir}")

    def _create_output_structure(self):
        """创建输出目录结构"""
        for split in ['train', 'val']:
            safe_mkdir(self.output_dir / "images" / split)
            safe_mkdir(self.output_dir / "labels" / split)
            if self.save_json:
                safe_mkdir(self.output_dir / "json_annotations" / split)

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
        # 使用统一的切图计算工具
        return TileCalculator.calculate_tiles(
            image_size=(h, w),
            tile_size=self.tile_size,
            overlap=self.overlap
        )

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
            # logger.info(f"🔤 '{text}' -> '{result}'")
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
        write_json(json_path, json_data, ensure_ascii=False, indent=2)

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


# 已删除 process_json_file 函数，使用 process_dataset() 统一处理

def process_dataset(
    input_source: str,
    image_dir: str = "images",
    output_base_dir: str = "datasets",
    final_output_dir: str = None,
    temp_dir: str = None,
    clean_temp: bool = True,
    tile_size: int = 640,
    overlap: int = 200,
    split_ratio: float = 0.8,
    min_area_ratio: float = 0.85,
    merge_manual_datasets: bool = False,
    manual_datasets_dir: str = "datasets",
) -> dict:
    """
    通用的数据集处理函数

    输入规则：
    - 单个JSON文件: input_source="annotations/红木.json" → 处理单个文件
    - 文件夹: input_source="annotations" → 批量处理文件夹下所有JSON
    - 逗号分隔: input_source="file1.json,file2.json" → 批量处理多个文件

    Args:
        input_source: 输入源（JSON文件/文件夹/逗号分隔列表）
        image_dir: 图片目录（默认: images）
        output_base_dir: 输出基础目录（默认: datasets）
        final_output_dir: 最终合并输出目录（仅在 merge_manual_datasets=True 时使用）
        temp_dir: 临时输出目录（仅在 merge_manual_datasets=True 时使用）
        clean_temp: 是否清理临时目录（仅在 merge_manual_datasets=True 时使用）
        tile_size: 切片大小
        overlap: 重叠区域大小
        split_ratio: 训练集比例
        min_area_ratio: 最小保留比例
        merge_manual_datasets: 是否合并手动标注数据集（批量模式时）

    Returns:
        统计信息字典
    """
    input_path = Path(input_source)
    image_dir = Path(image_dir)
    temp_dir = Path(temp_dir) if temp_dir else Path("datasets/temp_tiler_output")

    # 收集需要处理的JSON文件
    json_files = []

    # 处理单个JSON文件
    if input_path.is_file() and input_path.suffix.lower() == '.json':
        logger.info(f"📁 处理单个JSON文件: {input_path}")
        json_files = [input_path]

    # 处理文件夹
    elif input_path.is_dir():
        logger.info(f"📂 处理文件夹: {input_path}")
        json_files = list(input_path.glob('*.json'))
        if not json_files:
            logger.warning(f"⚠️  在 {input_path} 中未找到JSON文件")
            return {"error": "未找到JSON文件"}
        logger.info(f"🔍 找到 {len(json_files)} 个JSON文件")

    # 处理多个JSON文件列表（逗号分隔）
    elif ',' in input_source:
        logger.info("📚 处理多个JSON文件列表")
        for path_str in input_source.split(','):
            json_file = Path(path_str.strip())
            if json_file.is_file():
                json_files.append(json_file)
                logger.info(f"  📄 {json_file.name}")
            else:
                logger.error(f"  ❌ 文件不存在: {json_file}")

        if not json_files:
            logger.error("❌ 没有有效的JSON文件")
            return {"error": "没有有效的JSON文件"}
    else:
        logger.error(f"❌ 输入路径无效: {input_path}")
        logger.error("💡 请提供有效的JSON文件路径、文件夹路径或逗号分隔的多个文件路径")
        return {"error": "输入路径无效"}

    # ========== 处理JSON文件 ==========
    tilered_datasets = []
    results = {
        'processed': 0,
        'failed': 0,
        'train_tiles': 0,
        'val_tiles': 0,
        'kept_complete': 0,
        'skipped_incomplete': 0,
    }

    for json_file in sorted(json_files):
        json_stem = json_file.stem
        logger.info(f"\n📄 处理: {json_stem}")

        try:
            # 查找匹配的图片
            image_path = find_matching_image(json_stem, image_dir)

            # 构建配置
            if merge_manual_datasets:
                # 批量+合并模式：输出到临时目录
                output_dir = temp_dir / json_stem
            else:
                # 单独/批量模式：输出到独立目录
                output_dir = Path(output_base_dir) / json_stem

            config = {
                "image_path": str(image_path),
                "json_path": str(json_file),
                "output_dir": str(output_dir),
                "tile_size": tile_size,
                "overlap": overlap,
                "split_ratio": split_ratio,
                "min_val_tiles": 3,
                "class_names": ["booth"],
                "dataset_name": json_stem,
                "min_area_ratio": min_area_ratio,
                "keep_only_complete": True,
                "save_json": False,
            }

            # 创建切分器并执行
            tiler = Tiler(config)
            result = tiler.process()

            tilered_datasets.append(Path(result['output_dir']))

            # 累计统计
            results['processed'] += 1
            results['train_tiles'] += result['train_tiles']
            results['val_tiles'] += result['val_tiles']
            results['kept_complete'] += result['kept_complete']
            results['skipped_incomplete'] += result['skipped_incomplete']

            logger.info(f"✅ {json_stem} 完成: {result['train_tiles']} 训练切片, {result['val_tiles']} 验证切片")

        except FileNotFoundError as e:
            logger.error(f"❌ {json_stem} 跳过: {e}")
            results['failed'] += 1
            continue
        except Exception as e:
            logger.error(f"❌ {json_stem} 失败: {e}")
            results['failed'] += 1
            continue

    # ========== 合并手动标注数据集（可选） ==========
    if merge_manual_datasets:
        logger.info("\n" + "=" * 60)
        logger.info("🔗 合并手动标注数据集")
        logger.info("=" * 60)

        final_output_dir = Path(final_output_dir) if final_output_dir else Path("datasets/booth_final_merged")
        safe_mkdir(temp_dir)

        # 收集手动标注数据集
        manual_datasets_dir = Path(manual_datasets_dir)
        valid_datasets = []

        for dataset_dir in manual_datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            required_dirs = [
                dataset_dir / "images" / "train",
                dataset_dir / "images" / "val",
                dataset_dir / "labels" / "train",
                dataset_dir / "labels" / "val",
            ]

            if all(d.exists() for d in required_dirs):
                train_imgs = len(list((dataset_dir / "images" / "train").glob("*")))
                val_imgs = len(list((dataset_dir / "images" / "val").glob("*")))

                if train_imgs > 0 or val_imgs > 0:
                    valid_datasets.append(dataset_dir)
                    logger.info(f"✅ {dataset_dir.name}: {train_imgs} 训练, {val_imgs} 验证")

        # 合并所有数据集
        all_datasets = tilered_datasets + valid_datasets
        logger.info(f"📦 待合并: {len(all_datasets)} 个")

        final_train_img_dir = final_output_dir / "images" / "train"
        final_train_lbl_dir = final_output_dir / "labels" / "train"
        final_val_img_dir = final_output_dir / "images" / "val"
        final_val_lbl_dir = final_output_dir / "labels" / "val"

        for dir_path in [final_train_img_dir, final_train_lbl_dir, final_val_img_dir, final_val_lbl_dir]:
            safe_mkdir(dir_path)

        merge_stats = {
            'train_images': 0,
            'val_images': 0,
            'train_annotations': 0,
            'val_annotations': 0,
            'datasets_count': 0,
        }

        for dataset_dir in all_datasets:
            logger.info(f"\n🔗 合并: {dataset_dir.name}")
            dataset_prefix = f"{dataset_dir.name}_"

            # 训练集
            train_img_dir = dataset_dir / "images" / "train"
            train_lbl_dir = dataset_dir / "labels" / "train"

            if train_img_dir.exists():
                for img_file in train_img_dir.glob("*"):
                    if img_file.is_file():
                        new_name = f"{dataset_prefix}{img_file.name}"
                        shutil.copy2(img_file, final_train_img_dir / new_name)

                        label_file = train_lbl_dir / img_file.with_suffix('.txt').name
                        if label_file.exists():
                            shutil.copy2(label_file, final_train_lbl_dir / new_name)
                            merge_stats['train_annotations'] += 1

                merge_stats['train_images'] += len(list(train_img_dir.glob("*")))

            # 验证集
            val_img_dir = dataset_dir / "images" / "val"
            val_lbl_dir = dataset_dir / "labels" / "val"

            if val_img_dir.exists():
                for img_file in val_img_dir.glob("*"):
                    if img_file.is_file():
                        new_name = f"{dataset_prefix}{img_file.name}"
                        shutil.copy2(img_file, final_val_img_dir / new_name)

                        label_file = val_lbl_dir / img_file.with_suffix('.txt').name
                        if label_file.exists():
                            shutil.copy2(label_file, final_val_lbl_dir / new_name)
                            merge_stats['val_annotations'] += 1

                merge_stats['val_images'] += len(list(val_img_dir.glob("*")))

            merge_stats['datasets_count'] += 1

        # 生成 dataset.yaml
        path_str = str(final_output_dir.absolute())
        yaml_content = f"""# 最终合并数据集
path: {path_str}
train: images/train
val: images/val

names:
  0: booth
"""
        (final_output_dir / "dataset.yaml").write_text(yaml_content, encoding='utf-8')

        logger.info("\n" + "=" * 60)
        logger.info("📊 合并统计")
        logger.info("=" * 60)
        logger.info(f"合并数据集: {merge_stats['datasets_count']}")
        logger.info(f"训练集图片: {merge_stats['train_images']}")
        logger.info(f"验证集图片: {merge_stats['val_images']}")
        logger.info(f"训练集标注: {merge_stats['train_annotations']}")
        logger.info(f"验证集标注: {merge_stats['val_annotations']}")
        logger.info(f"输出: {final_output_dir}")
        logger.info("=" * 60)

        # 清理临时目录
        if clean_temp and temp_dir.exists():
            logger.info(f"\n🧹 清理临时目录: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                logger.info("✅ 临时目录已删除")
            except Exception as e:
                logger.warning(f"⚠️  清理临时目录失败: {e}")

        # 合并统计信息
        results.update(merge_stats)

    # ========== 打印最终统计 ==========
    if not merge_manual_datasets:
        logger.info("\n" + "=" * 60)
        logger.info("📊 处理统计")
        logger.info("=" * 60)
        logger.info(f"处理文件: {results['processed']}")
        logger.info(f"失败文件: {results['failed']}")
        logger.info(f"训练集切片: {results['train_tiles']}")
        logger.info(f"验证集切片: {results['val_tiles']}")
        logger.info(f"保留完整标注: {results['kept_complete']}")
        logger.info(f"跳过不完整: {results['skipped_incomplete']}")
        logger.info("=" * 60)

    return results


if __name__ == "__main__":
    # 模式1: 处理单个文件
    # process_dataset("annotations/红木.json")

    # 模式2: 批量处理文件夹
    # process_dataset("annotations")

    # 模式3: 批量处理 + 合并手动标注数据集
    process_dataset(
        input_source="annotations/红木.json",
        merge_manual_datasets=True,
        manual_datasets_dir="datasets/manual_booth_annotations",
        final_output_dir="datasets/booth_final_merged",
        clean_temp=True,
        tile_size=640,
        overlap=200,
    )

