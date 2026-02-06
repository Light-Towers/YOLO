"""
项目常量定义
集中管理所有魔法数字和配置常量
"""
from dataclasses import dataclass
from typing import List


@dataclass
class ImageConstants:
    """图像相关常量"""
    DEFAULT_IMAGE_SIZE: int = 640
    MAX_IMAGE_SIZE: int = 4096
    MIN_IMAGE_SIZE: int = 32
    SUPPORTED_FORMATS: List[str] = None

    def __post_init__(self):
        if self.SUPPORTED_FORMATS is None:
            self.SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']


@dataclass
class TrainingConstants:
    """训练相关常量"""
    DEFAULT_EPOCHS: int = 300
    DEFAULT_PATIENCE: int = 50
    DEFAULT_BATCH: float = 0.9
    DEFAULT_LR: float = 0.01
    DEFAULT_MOMENTUM: float = 0.937
    DEFAULT_WEIGHT_DECAY: float = 0.0005
    WARMUP_EPOCHS: int = 3

    # 数据增强参数
    DEFAULT_DEGREES: float = 15.0
    DEFAULT_TRANSLATE: float = 0.1
    DEFAULT_SCALE: float = 0.5
    DEFAULT_SHEAR: float = 0.0
    DEFAULT_PERSPECTIVE: float = 0.001
    DEFAULT_FLIPUD: float = 0.0
    DEFAULT_FLIPLR: float = 0.5
    DEFAULT_MOSAIC: float = 1.0
    DEFAULT_MIXUP: float = 0.1
    DEFAULT_COPY_PASTE: float = 0.0


@dataclass
class InferenceConstants:
    """推理相关常量"""
    DEFAULT_CONFIDENCE: float = 0.7
    DEFAULT_IOU: float = 0.2
    DEFAULT_SLICE_SIZE: int = 640
    DEFAULT_OVERLAP_RATIO: float = 0.5
    DEFAULT_MATCH_THRESHOLD: float = 0.5


@dataclass
class DatasetConstants:
    """数据集相关常量"""
    DEFAULT_TRAIN_RATIO: float = 0.8
    DEFAULT_MIN_VAL_TILES: int = 2
    DEFAULT_TILE_SIZE: int = 640
    DEFAULT_OVERLAP: int = 200
    DEFAULT_MIN_AREA_RATIO: float = 0.9


@dataclass
class LogConstants:
    """日志相关常量"""
    DEFAULT_LOG_LEVEL: str = "INFO"
    LOG_FORMAT_COLOR: str = "colored"
    LOG_FORMAT_PLAIN: str = "plain"
    LOG_FORMAT_JSON: str = "json"


# 全局常量实例
IMAGE_CONSTANTS = ImageConstants()
TRAINING_CONSTANTS = TrainingConstants()
INFERENCE_CONSTANTS = InferenceConstants()
DATASET_CONSTANTS = DatasetConstants()
LOG_CONSTANTS = LogConstants()
