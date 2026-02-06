"""
配置管理模块
支持从YAML文件加载配置
"""
import yaml
from pathlib import Path
from typing import Any, Optional, Dict
from dataclasses import dataclass, field

from src.core.exceptions import ConfigurationError


@dataclass
class ProjectConfig:
    """项目配置"""
    name: str = "yolo-booth-detection"
    version: str = "0.1.0"


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str = "booth_seg"
    root_dir: str = "datasets"
    train_ratio: float = 0.8
    min_val_tiles: int = 2


@dataclass
class TrainingConfig:
    """训练配置"""
    models: list[str] = field(default_factory=list)
    epochs: int = 300
    patience: int = 50
    batch: float = 0.9
    imgsz: int = 640
    device: str = "auto"
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    degrees: float = 15.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.001
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.0
    dropout: float = 0.0
    cos_lr: bool = True
    single_cls: bool = True
    overlap_mask: bool = False


@dataclass
class InferenceConfig:
    """推理配置"""
    confidence_threshold: float = 0.7
    iou_threshold: float = 0.2
    hide_labels: bool = True
    hide_conf: bool = True
    rect_th: int = 2

    # SAHI配置
    sahi_enabled: bool = True
    sahi_slice_height: int = 640
    sahi_slice_width: int = 640
    sahi_overlap_height_ratio: float = 0.5
    sahi_overlap_width_ratio: float = 0.5
    sahi_postprocess_type: str = "NMS"
    sahi_postprocess_match_metric: str = "IOS"
    sahi_postprocess_match_threshold: float = 0.5


@dataclass
class PathsConfig:
    """路径配置"""
    models_dir: str = "models"
    datasets_dir: str = "datasets"
    outputs_dir: str = "output"
    logs_dir: str = "logs"
    images_dir: str = "images"


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "colored"
    log_to_file: bool = True
    log_dir: str = "logs"


@dataclass
class Config:
    """完整配置"""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> 'Config':
        """从YAML文件加载配置"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise ConfigurationError(f"配置文件不存在: {yaml_path}")

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            return cls(
                project=ProjectConfig(**data.get('project', {})),
                dataset=DatasetConfig(**data.get('dataset', {})),
                training=TrainingConfig(**data.get('training', {})),
                inference=InferenceConfig(**data.get('inference', {})),
                paths=PathsConfig(**data.get('paths', {})),
                logging=LoggingConfig(**data.get('logging', {})),
            )
        except Exception as e:
            raise ConfigurationError(f"加载配置文件失败: {e}") from e

    def to_yaml(self, yaml_path: Path | str) -> None:
        """保存配置到YAML文件"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'project': self.project.__dict__,
            'dataset': self.dataset.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'paths': self.paths.__dict__,
            'logging': self.logging.__dict__,
        }

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_config(config_path: Optional[Path | str] = None) -> Config:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，None则使用默认路径

    Returns:
        Config对象
    """
    if config_path is None:
        # 尝试查找默认配置文件
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "default.yaml"

    return Config.from_yaml(config_path)
