"""
pytest 配置文件
定义测试共享的 fixture
"""
import pytest
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """创建测试用的示例图片"""
    # 创建 1000x1000 的白色图片
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def sample_annotation(tmp_path: Path) -> Path:
    """创建测试用的 LabelMe 标注文件"""
    annotation = {
        "version": "5.0.1",
        "shapes": [
            {
                "label": "booth",
                "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
                "shape_type": "polygon"
            },
            {
                "label": "booth",
                "points": [[300, 300], [400, 300], [400, 400], [300, 400]],
                "shape_type": "polygon"
            }
        ],
        "imageHeight": 1000,
        "imageWidth": 1000
    }

    json_path = tmp_path / "test_annotation.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    return json_path


@pytest.fixture
def sample_config(tmp_path: Path) -> Dict[str, Any]:
    """创建测试用的配置字典"""
    return {
        "image_path": str(tmp_path / "test_image.jpg"),
        "json_path": str(tmp_path / "test_annotation.json"),
        "output_dir": str(tmp_path / "output"),
        "tile_size": 640,
        "overlap": 200,
        "split_ratio": 0.8,
        "min_val_tiles": 2,
    }


@pytest.fixture
def sample_dataset_yaml(tmp_path: Path) -> Path:
    """创建测试用的数据集配置文件"""
    yaml_content = """
path: {data_path}
train: images/train
val: images/val
names:
  0: booth
"""
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(yaml_content.format(data_path=str(tmp_path)))
    return yaml_path


@pytest.fixture
def mock_yolo_model():
    """模拟 YOLO 模型"""
    from unittest.mock import Mock

    mock_model = Mock()
    mock_model.train.return_value = Mock()
    mock_model.predict.return_value = []
    return mock_model


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """创建临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir
