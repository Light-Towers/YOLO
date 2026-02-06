"""
测试配置加载模块
"""
import pytest
from pathlib import Path

from src.core.config import Config, ProjectConfig, TrainingConfig, DatasetConfig
from src.core.exceptions import ConfigurationError


class TestConfig:
    """配置类测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = Config()
        assert config.project.name == "yolo-booth-detection"
        assert config.training.epochs == 300
        assert config.dataset.train_ratio == 0.8

    def test_project_config(self):
        """测试项目配置"""
        project = ProjectConfig(name="test-project", version="1.0.0")
        assert project.name == "test-project"
        assert project.version == "1.0.0"

    def test_training_config(self):
        """测试训练配置"""
        training = TrainingConfig(epochs=100, patience=30)
        assert training.epochs == 100
        assert training.patience == 30

    def test_dataset_config(self):
        """测试数据集配置"""
        dataset = DatasetConfig(name="test-dataset", train_ratio=0.9)
        assert dataset.name == "test-dataset"
        assert dataset.train_ratio == 0.9

    def test_load_config_from_yaml(self, tmp_path):
        """测试从 YAML 加载配置"""
        yaml_content = """
project:
  name: "test-project"
  version: "1.0.0"

dataset:
  name: "test-dataset"
  train_ratio: 0.9

training:
  epochs: 100
  models:
    - "yolov8s-obb.pt"
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        config = Config.from_yaml(yaml_path)
        assert config.project.name == "test-project"
        assert config.dataset.name == "test-dataset"
        assert config.training.epochs == 100
        assert "yolov8s-obb.pt" in config.training.models

    def test_load_nonexistent_yaml(self, tmp_path):
        """测试加载不存在的 YAML 文件"""
        yaml_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(ConfigurationError):
            Config.from_yaml(yaml_path)

    def test_save_config_to_yaml(self, tmp_path):
        """测试保存配置到 YAML"""
        config = Config()
        config.training.epochs = 200

        yaml_path = tmp_path / "output_config.yaml"
        config.to_yaml(yaml_path)

        loaded_config = Config.from_yaml(yaml_path)
        assert loaded_config.training.epochs == 200

    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = Config()
        project_dict = config.project.__dict__

        assert "name" in project_dict
        assert "version" in project_dict
