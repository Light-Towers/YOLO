"""
切分器 - 只保留完整的展位标注，避免形状被切割
处理原始图片和json文件, 生成数据集
"""
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime

import cv2
import shapely.affinity as affinity
from pypinyin import lazy_pinyin
from shapely.geometry import Polygon, box

from src.core import DATASET_CONSTANTS

# 导入工程化工具
from src.utils import (
    ensure_absolute,
    get_logger,
    get_project_root,
    read_json,
    safe_mkdir,
    write_json,
)
from src.utils.image_tile_utils import TileCalculator

logger = get_logger('dataset_tiler')


def _convert_to_pinyin(name: str) -> str:
    """将中文名转换为拼音"""
    return "".join(lazy_pinyin(name))

class Tiler:
    """
    YOLO数据集切分器

    关键修复:
    1. 只保留完整在切片内的展位标注（不切割多边形）
    2. 增大overlap确保每个展位至少在一个切片中是完整的
    3. 可选：对于部分在切片内的展位，使用其完整边界框
    """

    def __init__(self, config: Dict[str, Any]):
        self.image_path = Path(config["image_path"])
        self.json_path = Path(config["json_path"])
        self.output_dir = Path(config["output_dir"])
        self.tile_size = config.get("tile_size", DATASET_CONSTANTS.DEFAULT_TILE_SIZE)
        self.overlap = config.get("overlap", DATASET_CONSTANTS.DEFAULT_OVERLAP)
        self.class_names = config.get("class_names", ["booth"])
        self.min_area_ratio = config.get("min_area_ratio", DATASET_CONSTANTS.DEFAULT_MIN_AREA_RATIO)
        self.keep_only_complete = config.get("keep_only_complete", True)
        self.save_json = config.get("save_json", False)

        safe_mkdir(self.output_dir)

        self.img = cv2.imread(str(self.image_path))
        if self.img is None:
            raise ValueError(f"无法读取图像: {self.image_path}")

        self.labelme_data = read_json(self.json_path)

        logger.info(f"🖼️  原图尺寸: {self.img.shape[1]}x{self.img.shape[0]}")
        logger.info(f"🏷️  标注对象数量: {len(self.labelme_data['shapes'])}")
        logger.info(f"📁 输出目录: {self.output_dir}")

    def _get_all_tiles(self) -> List[Tuple[int, int, int, int, int]]:
        """获取所有切片位置"""
        h, w = self.img.shape[:2]
        # 使用统一的切图计算工具
        return TileCalculator.calculate_tiles(
            image_size=(h, w),
            tile_size=self.tile_size,
            overlap=self.overlap
        )

    def _is_polygon_complete_in_tile(self, poly: Polygon, tile_box: box) -> bool:
        """检查多边形是否完整在切片内"""
        if not poly.intersects(tile_box):
            return False
        
        # 计算交集面积比例
        intersection = poly.intersection(tile_box)
        area_ratio = intersection.area / poly.area if poly.area > 0 else 0
        
        return area_ratio >= self.min_area_ratio

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

        stats = {'total': 0, 'kept': 0, 'skipped': 0}

        for tile_id, x, y, x_end, y_end in all_tiles:
            self._process_tile(tile_id, x, y, x_end, y_end, stats)

        logger.info(f"📊 切分完成: {len(all_tiles)} 切片, 保留 {stats['kept']} 个标注")

        return {
            'output_dir': str(self.output_dir),
            'total_tiles': len(all_tiles),
            'kept': stats['kept'],
            'skipped': stats['skipped']
        }

    def _process_tile(self, tile_id: int, x: int, y: int, x_end: int, y_end: int, stats: dict) -> dict:
        """处理单个切片 - 根据是否有标注分别保存到 annotated/ 或 background/ 目录"""
        tile_w, tile_h = x_end - x, y_end - y
        tile_img = self.img[y:y_end, x:x_end]

        tile_name = f"{_convert_to_pinyin(self.image_path.stem)}_tile_{tile_id:04d}.png"

        # 处理标注
        annotations = []
        for shape in self.labelme_data["shapes"]:
            if shape["shape_type"] not in ("polygon", "rotation"):
                continue
            stats['total'] += 1
            ann = self._convert_annotation_fixed(shape, x, y, tile_w, tile_h)
            if ann:
                annotations.append(ann)
                stats['kept'] += 1
            else:
                stats['skipped'] += 1

        # 根据是否有标注选择子目录
        has_annotation = len(annotations) > 0
        target_subdir = "annotated" if has_annotation else "background"
        target_dir = self.output_dir / target_subdir
        safe_mkdir(target_dir)

        # 保存图像到对应子目录
        img_path = target_dir / tile_name
        cv2.imwrite(str(img_path), tile_img)

        # 如果有标注且需要保存JSON，则保存
        if self.save_json and has_annotation:
            self._save_json_annotation(tile_name, tile_w, tile_h, annotations, target_dir)

        return {'name': tile_name, 'annotations': len(annotations), 'subdir': target_subdir}

    def _save_json_annotation(self, tile_name: str, tile_w: int, tile_h: int, annotations: List[dict], target_dir: Path = None):
        """保存JSON格式标注（用于标注工具检查）"""

        # 如果未指定目标目录，使用默认输出目录
        if target_dir is None:
            target_dir = self.output_dir

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

        # 保存JSON文件到目标目录
        write_json(target_dir / tile_name.replace(".png", ".json"), json_data, indent=2)


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
    max_background_ratio: float = 0.3,  # 背景图在训练集中的最大比例
    min_background_per_source: int = 1,  # 每个原始图片来源至少保留的背景图数量
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
        manual_datasets_dir: 手动标注数据目录，支持单个目录或多个目录（逗号分隔，默认: datasets）
        max_background_ratio: 背景图在训练集中的最大比例（默认0.3=30%），避免背景图过多
        min_background_per_source: 每个原始图片来源至少保留的背景图数量（默认1），保证来源多样性

    Returns:
        统计信息字典
    """
    # 获取项目根目录，用于将相对路径转为绝对路径
    project_root = get_project_root()

    input_path = ensure_absolute(input_source, project_root)
    image_dir = ensure_absolute(image_dir, project_root)
    temp_dir = ensure_absolute(temp_dir, project_root) if temp_dir else project_root / "datasets" / "temp_tiler_output"

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

    # ========== 步骤1: 切分 input_source 中的 JSON 文件（生成切片图片 + JSON） ==========
    # 切分后的数据放到 datasets/tmp/tiling_xx 下
    tmp_base_dir = project_root / "datasets" / "tmp"
    safe_mkdir(tmp_base_dir)

    results = {'processed': 0, 'failed': 0, 'total_tiles': 0, 'kept': 0}

    logger.info("\n" + "=" * 60)
    logger.info("📋 步骤1: 切分 JSON 文件")
    logger.info("=" * 60)

    for json_file in sorted(json_files):
        json_stem = json_file.stem
        logger.info(f"\n📄 处理: {json_stem}")

        try:
            # 查找匹配的图片
            image_path = find_matching_image(json_stem, image_dir)

            # 切分输出目录：datasets/tmp/tiling_xx（xx为拼音名）
            pinyin_name = _convert_to_pinyin(json_stem)
            output_dir = tmp_base_dir / f"tiling_{pinyin_name}"
            safe_mkdir(output_dir)

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
                "save_json": True,  # 保存切片后的 JSON
            }

            # 创建切分器并执行
            tiler = Tiler(config)
            result = tiler.process()

            # 累计统计
            results['processed'] += 1
            results['total_tiles'] += result['total_tiles']
            results['kept'] += result['kept']

            logger.info(f"✅ {json_stem} 完成: {result['total_tiles']} 切片")

        except FileNotFoundError as e:
            logger.error(f"❌ {json_stem} 跳过: {e}")
            results['failed'] += 1
            continue
        except Exception as e:
            logger.error(f"❌ {json_stem} 失败: {e}")
            results['failed'] += 1
            continue

    # 统计切分结果
    tiling_dirs = list(tmp_base_dir.glob('tiling_*'))
    # 统计所有子目录中的PNG文件（annotated和background目录）
    total_png = sum(
        len(list((d / 'annotated').glob('*.png'))) + len(list((d / 'background').glob('*.png')))
        for d in tiling_dirs
    )
    logger.info(f"\n📦 步骤1完成: {len(tiling_dirs)} 个目录, {total_png} 个切片")

    # ========== 步骤2: 合并并分类（annotated/background） ==========
    if merge_manual_datasets:
        logger.info("\n" + "=" * 60)
        logger.info("📋 步骤2: 合并并分类数据")
        logger.info("=" * 60)

        # 创建 mix_tiling 目录用于分类存储
        mix_dir = tmp_base_dir / "mix_tiling"
        mix_annotated_dir = mix_dir / "annotated"
        mix_background_dir = mix_dir / "background"
        safe_mkdir(mix_annotated_dir)
        safe_mkdir(mix_background_dir)

        # 1. 处理 tiling_* 目录中的切分数据（数据已在切图时分类好）
        logger.info("📂 合并切分数据...")
        for tiling_dir in tiling_dirs:
            # 从 annotated 子目录复制有标注的数据
            annotated_dir = tiling_dir / 'annotated'
            if annotated_dir.exists():
                for json_file in annotated_dir.glob('*.json'):
                    try:
                        json_stem = json_file.stem
                        # 查找匹配图片
                        img_file = None
                        for ext in ['.png', '.jpg', '.jpeg']:
                            candidate = annotated_dir / f"{json_stem}{ext}"
                            if candidate.exists():
                                img_file = candidate
                                break

                        if not img_file:
                            logger.warning(f"⚠️  跳过 {json_stem}: 找不到匹配图片")
                            continue

                        # 复制到 annotated 目标目录
                        shutil.copy2(json_file, mix_annotated_dir)
                        shutil.copy2(img_file, mix_annotated_dir)
                    except Exception as e:
                        logger.error(f"❌ 复制失败 {json_file.name}: {e}")
                        continue

            # 从 background 子目录复制背景图
            background_dir = tiling_dir / 'background'
            if background_dir.exists():
                for png_file in background_dir.glob('*.png'):
                    try:
                        # 背景图只有PNG，没有JSON
                        shutil.copy2(png_file, mix_background_dir)
                    except Exception as e:
                        logger.error(f"❌ 复制失败 {png_file.name}: {e}")
                        continue

        # 2. 合并 manual_datasets_dir 中的手动标注数据（全部视为有标注）
        if manual_datasets_dir:
            # 支持多个目录（逗号分隔）
            manual_dirs = []
            for dir_str in manual_datasets_dir.split(','):
                dir_path = ensure_absolute(dir_str.strip(), project_root)
                if dir_path.is_dir():
                    manual_dirs.append(dir_path)

            if manual_dirs:
                for manual_dir in manual_dirs:
                    logger.info(f"📂 合并手动标注数据: {manual_dir}")
                    for json_file in manual_dir.glob('*.json'):
                        shutil.copy2(json_file, mix_annotated_dir)
                        json_stem = json_file.stem
                        for ext in ['.png', '.jpg', '.jpeg']:
                            img_file = manual_dir / f"{json_stem}{ext}"
                            if img_file.exists():
                                shutil.copy2(img_file, mix_annotated_dir)
                                break

        # 统计分类结果
        annotated_count = len(list(mix_annotated_dir.glob('*.json')))
        background_count = len(list(mix_background_dir.glob('*.png')))  # 背景图只有PNG，没有JSON
        logger.info(f"\n📦 步骤2完成:")
        logger.info(f"   ✅ 有标注(annotated): {annotated_count} 个")
        logger.info(f"   ⚪ 背景图(background): {background_count} 个")

        # ========== 步骤3: 将 tmp 目录转换为 YOLO 格式数据集 ==========
        logger.info("\n" + "=" * 60)
        logger.info("📋 步骤3: 转换为 YOLO 格式数据集")
        logger.info("=" * 60)

        final_output_dir = ensure_absolute(final_output_dir, project_root) if final_output_dir else project_root / "datasets" / "booth_final_merged"
        final_train_img_dir = final_output_dir / "images" / "train"
        final_train_lbl_dir = final_output_dir / "labels" / "train"
        final_val_img_dir = final_output_dir / "images" / "val"
        final_val_lbl_dir = final_output_dir / "labels" / "val"

        # 清理已存在的输出目录（避免重复执行时图片累积）
        if final_output_dir.exists():
            logger.info(f"🧹 清理已存在的输出目录: {final_output_dir}")
            shutil.rmtree(final_output_dir)

        for dir_path in [final_train_img_dir, final_train_lbl_dir, final_val_img_dir, final_val_lbl_dir]:
            safe_mkdir(dir_path)

        # 处理 mix_tiling 目录中的所有 JSON（从分类后的目录读取）
        json_count = 0
        train_count = 0
        val_count = 0
        
        # 先处理 annotated 目录（有标注的按 split_ratio 分配）
        for json_file in mix_annotated_dir.glob('*.json'):
            try:
                json_stem = json_file.stem
                json_dir = json_file.parent
                # 查找匹配图片（在同目录）
                image_file = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    candidate = json_dir / f"{json_stem}{ext}"
                    if candidate.exists():
                        image_file = candidate
                        break

                if not image_file:
                    logger.warning(f"⚠️  跳过 {json_stem}: 找不到匹配图片")
                    continue

                # 读取 JSON 并转换为 YOLO 格式
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                img_width = data.get('imageWidth', data.get('image_width', 0))
                img_height = data.get('imageHeight', data.get('image_height', 0))

                if img_width == 0 or img_height == 0:
                    # 尝试从图片获取尺寸
                    from PIL import Image
                    with Image.open(image_file) as img:
                        img_width, img_height = img.size

                yolo_annotations = []
                for shape in data.get('shapes', []):
                    label = shape.get('label', 'booth')
                    points = shape.get('points', [])
                    if len(points) >= 4:
                        # 将多边形转换为归一化坐标
                        x_coords = [p[0] / img_width for p in points]
                        y_coords = [p[1] / img_height for p in points]
                        yolo_ann = '0 ' + ' '.join([f"{x:.6f} {y:.6f}" for x, y in zip(x_coords, y_coords)])
                        yolo_annotations.append(yolo_ann)

                # 准备标注内容
                label_content = '\n'.join(yolo_annotations) + '\n' if yolo_annotations else ''

                # annotated 目录：有标注的按 split_ratio 分配
                is_train = random.random() < split_ratio

                try:
                    if is_train:
                        shutil.copy2(image_file, final_train_img_dir / image_file.name)
                        (final_train_lbl_dir / f"{json_stem}.txt").write_text(label_content, encoding='utf-8')
                        train_count += 1
                    else:
                        shutil.copy2(image_file, final_val_img_dir / image_file.name)
                        (final_val_lbl_dir / f"{json_stem}.txt").write_text(label_content, encoding='utf-8')
                        val_count += 1
                    json_count += 1
                except Exception as copy_err:
                    logger.error(f"❌ 复制失败 {json_stem}: {copy_err}")
                    continue

            except Exception as e:
                logger.error(f"❌ 转换失败 {json_file.name}: {e}")
                continue

        # 处理 background 目录（智能筛选，保证原始图片来源多样性）
        logger.info("📂 处理背景图（智能筛选）...")

        # 计算应该保留的背景图数量
        annotated_train_count = train_count
        max_background_count = int(annotated_train_count * max_background_ratio / (1 - max_background_ratio))

        # 收集所有背景图并按原始图片分组
        # 背景图只有PNG文件，格式: {original_name}_tile_{tile_id}.png
        background_files = list(mix_background_dir.glob('*.png'))
        
        # 按原始图片来源分组
        source_groups = defaultdict(list)
        for png_file in background_files:
            # 解析原始图片名称 (如 hongmu_tile_0001.png -> hongmu)
            stem = png_file.stem
            if '_tile_' in stem:
                source_name = stem.rsplit('_tile_', 1)[0]
            else:
                source_name = 'unknown'
            source_groups[source_name].append(png_file)
        
        # 智能筛选策略：
        # 1. 优先从样本量多的来源组中选取（数据更丰富的来源）
        # 2. 使用轮询(round-robin)方式从各组选取，保证来源多样性
        # 3. 确保每个来源至少有 min_background_per_source 个样本
        
        selected_backgrounds = []
        
        if len(background_files) <= max_background_count:
            # 背景图数量未超限，全部使用
            selected_backgrounds = background_files
            logger.info(f"   ✅ 背景图总数 {len(background_files)} 未超限 ({max_background_count})，全部使用")
        else:
            # 需要筛选：按来源丰富度排序，优先从多样性的来源选取
            logger.info(f"   🔍 背景图共 {len(background_files)} 个，来自 {len(source_groups)} 个原始图片")
            logger.info(f"   ⚠️ 限制为 {max_background_count} 个 (比例 {max_background_ratio:.0%})")
            logger.info(f"   📌 每个来源最少保留: {min_background_per_source} 个")
            
            # 按组大小降序排序（优先从样本多的组选取）
            sorted_groups = sorted(source_groups.items(), key=lambda x: len(x[1]), reverse=True)
            
            # 第一阶段：确保每个来源至少有 min_background_per_source 个样本
            for source_name, files in sorted_groups:
                samples_to_take = min(min_background_per_source, len(files))
                selected_backgrounds.extend(files[:samples_to_take])
            
            # 第二阶段：如果还有配额，轮询补充
            if len(selected_backgrounds) < max_background_count:
                remaining_quota = max_background_count - len(selected_backgrounds)
                group_indices = {name: min_background_per_source for name, _ in sorted_groups}
                group_names = [name for name, files in sorted_groups if len(files) > min_background_per_source]
                
                while remaining_quota > 0 and group_names:
                    for source_name in list(group_names):
                        if remaining_quota <= 0:
                            break
                        
                        group = source_groups[source_name]
                        idx = group_indices[source_name]
                        
                        if idx < len(group):
                            selected_backgrounds.append(group[idx])
                            group_indices[source_name] += 1
                            remaining_quota -= 1
                        else:
                            group_names.remove(source_name)
            
            # 输出选取统计
            logger.info(f"   📊 智能筛选完成，选中 {len(selected_backgrounds)} 个背景图")
            for source_name, files in sorted_groups[:5]:  # 显示前5个来源
                # 计算该来源被选中的数量
                selected_count = sum(1 for f in selected_backgrounds
                                   if f.stem.startswith(source_name))
                total_count = len(files)
                if selected_count > 0:
                    logger.info(f"      • {source_name}: {selected_count}/{total_count}")
        
        background_files = selected_backgrounds

        for png_file in background_files:
            try:
                json_stem = png_file.stem

                # 背景图：空标注，强制放训练集
                shutil.copy2(png_file, final_train_img_dir / png_file.name)
                (final_train_lbl_dir / f"{json_stem}.txt").write_text('', encoding='utf-8')
                train_count += 1
                json_count += 1

            except Exception as e:
                logger.error(f"❌ 处理背景图失败 {png_file.name}: {e}")
                continue

        logger.info(f"\n📦 步骤3完成: 转换 {json_count} 个 JSON 到 YOLO 格式")

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

        # 统计有内容的标注文件数（排除空文件）
        train_labels_with_content = sum(1 for f in final_train_lbl_dir.glob('*.txt') if f.stat().st_size > 0)
        val_labels_with_content = sum(1 for f in final_val_lbl_dir.glob('*.txt') if f.stat().st_size > 0)

        logger.info("\n" + "=" * 60)
        logger.info("📊 最终统计")
        logger.info("=" * 60)
        logger.info(f"训练集: {len(list(final_train_img_dir.glob('*')))} 图片, {len(list(final_train_lbl_dir.glob('*.txt')))} 标注文件 ({train_labels_with_content} 有内容)")
        logger.info(f"验证集: {len(list(final_val_img_dir.glob('*')))} 图片, {len(list(final_val_lbl_dir.glob('*.txt')))} 标注文件 ({val_labels_with_content} 有内容)")
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

    # ========== 打印最终统计 ==========
    if not merge_manual_datasets:
        logger.info("\n" + "=" * 60)
        logger.info("📊 处理统计")
        logger.info("=" * 60)
        logger.info(f"处理文件: {results['processed']}, 失败: {results['failed']}")
        logger.info(f"总切片: {results['total_tiles']}, 保留标注: {results['kept']}")
        logger.info("=" * 60)

    return results


if __name__ == "__main__":
    # # 模式1: 处理单个文件
    # process_dataset("annotations/2024-28届宠物水族展.json", tile_size=1024,)

    # 模式2: 批量处理文件夹
    # process_dataset("annotations/红木.json,annotations/11届猪业.json")

    # 模式3: 批量处理 + 合并手动标注数据集
    process_dataset(
        input_source="annotations/",
        merge_manual_datasets=True,
        manual_datasets_dir="datasets/manual_booth_annotations, datasets/new_20260226_1530",  # 支持多个目录（逗号分隔）
        final_output_dir=f"datasets/booth_{datetime.now().strftime('%Y%m%d_%H%M')}",
        clean_temp=True,
        tile_size=1024,
        overlap=500,
        max_background_ratio=0.3,  # 背景图最多占训练集的30%
        min_background_per_source=2,  # 每个原始图片至少保留2个背景切片
    )