#!/usr/bin/env python3
"""
训练预测流水线 - 精简版
核心逻辑：
- train: 是否执行训练（有指定模型就用指定模型，没有就用默认预训练模型）
- predict: 是否执行预测（有指定模型就用指定模型，没有就用刚训练好的/注册表中的）
"""
from ultralytics import YOLO
import os
from pathlib import Path
import re
import torch
import json
from datetime import datetime

# 导入工程化工具
from src.utils import (
    get_project_root,
    get_logger,
    get_image_files,
    safe_mkdir,
)
from src.core import TRAINING_CONSTANTS

# 导入train模块中的函数
from train import train_model, get_model_path, update_dataset_path

# 导入predict_sahi模块
from predict_sahi import start_predict

logger = get_logger('train_predict_pipeline')


# ========== 模型注册表管理 ==========

def get_registry_path(project_dir):
    """获取模型注册文件路径"""
    return project_dir / 'output' / 'models' / '.model'


def save_model_registry(project_dir, model_paths):
    """保存模型信息到注册表"""
    registry_path = get_registry_path(project_dir)
    safe_mkdir(registry_path.parent)
    
    registry_data = {
        'time': datetime.now().isoformat(),
        'models': [
            {'path': str(path), 'name': Path(path).name}
            for path, _ in model_paths
        ]
    }
    
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"模型注册表已更新: {registry_path}")


def load_model_registry(project_dir):
    """从注册表加载模型信息"""
    registry_path = get_registry_path(project_dir)
    
    if not registry_path.exists():
        return None
    
    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        models = data.get('models', [])
        return [(m['path'], m['name']) for m in models]
    except Exception as e:
        logger.error(f"读取注册表失败: {e}")
        return None


# ========== 配置类 ==========

class PipelineConfig:
    """流水线配置"""
    def __init__(self, project_dir):
        self.project_dir = project_dir
        
        # 基础配置
        self.dataset_name = None
        self.exp_name = None
        self.epochs = 100
        self.prediction_images = []
        
        # 开关
        self.do_train = False
        self.do_predict = False
        
        # 模型配置（None表示使用默认逻辑）
        self.predict_models = None    # 预测用的模型路径列表


# ========== 数据集管理 ==========

class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.dataset_root = config.project_dir / 'datasets' / config.dataset_name
        self.yaml_path = self.dataset_root / 'dataset.yaml'
    
    def prepare(self):
        """准备数据集"""
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"数据集不存在: {self.dataset_root}")
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"数据集配置不存在: {self.yaml_path}")
        
        update_dataset_path(self.yaml_path, self.dataset_root)
        logger.info(f"数据集准备完成: {self.config.dataset_name}")
    
    def get_yaml_path(self):
        return self.yaml_path


# ========== 训练管理 ==========

class Trainer:
    def __init__(self, config):
        self.config = config
        self.trained_models = []
    
    def get_train_models(self, model_names):
        """
        获取要训练的模型列表
        
        model_names: 模型名称列表
            - 如果是路径（包含/或\），直接使用
            - 否则，从 models/ 目录查找
        """
        if not model_names:
            raise ValueError("未指定训练模型，请提供 model_names")
        
        models = []
        for name in model_names:
            if '/' in name or '\\' in name or Path(name).is_absolute():
                # 是路径，直接使用
                path = Path(name)
                models.append((path, path.name))
            else:
                # 是文件名，从 models/ 目录查找
                path = get_model_path(name, self.config.project_dir)
                models.append((path, name))
        
        logger.info(f"训练模型列表: {[m[1] for m in models]}")
        return models
    
    def train(self, model_names=None):
        """执行训练"""
        dataset = DatasetManager(self.config)
        dataset.prepare()
        
        models = self.get_train_models(model_names or [])
        
        logger.info(f"{'='*50}")
        logger.info(f"开始训练 {len(models)} 个模型")
        logger.info(f"{'='*50}")
        
        for model_path, model_name in models:
            result = self._train_one(model_path, model_name, dataset)
            if result:
                self.trained_models.append((result, model_name))
        
        logger.info(f"{'='*50}")
        logger.info(f"训练完成: {len(self.trained_models)}/{len(models)}")
        logger.info(f"{'='*50}")
        
        return self.trained_models
    
    def _train_one(self, model_path, model_name, dataset):
        """训练单个模型"""
        logger.info(f"训练模型: {model_name}")
        
        if not model_path.exists():
            logger.warning(f"模型未找到: {model_path}，将自动下载")
        
        best_path = train_model(
            model_path,
            dataset.get_yaml_path(),
            self.config.project_dir,
            self.config.exp_name,
            self.config.dataset_name,
            epochs=self.config.epochs
        )
        
        return best_path


# ========== 预测管理 ==========

class Predictor:
    def __init__(self, config):
        self.config = config
    
    def get_predict_models(self, trained_models=None):
        """获取用于预测的模型列表"""
        # 优先级: 指定模型 > 刚训练的模型 > 注册表模型
        
        if self.config.predict_models:
            logger.info("使用指定的模型进行预测")
            return [(p, Path(p).name) for p in self.config.predict_models]
        
        if trained_models:
            logger.info("使用刚训练好的模型进行预测")
            return trained_models
        
        registry_models = load_model_registry(self.config.project_dir)
        if registry_models:
            logger.info("使用注册表中的模型进行预测")
            return registry_models
        
        return None
    
    def get_images(self):
        """获取所有待预测图片"""
        images = []
        for source in self.config.prediction_images:
            source_path = Path(source)
            if source_path.is_file():
                images.append(str(source_path))
            elif source_path.is_dir():
                images.extend(str(p) for p in get_image_files(source_path, recursive=True))
        return sorted(set(images))
    
    def predict(self, models):
        """执行预测"""
        images = self.get_images()
        
        if not images:
            logger.warning("没有找到待预测图片")
            return
        
        logger.info(f"找到 {len(images)} 张待预测图片")
        logger.info(f"{'='*50}")
        logger.info("开始预测")
        logger.info(f"{'='*50}")
        
        for model_path, model_name in models:
            self._predict_one(model_path, model_name, images)
    
    def _predict_one(self, model_path, model_name, images):
        """使用单个模型预测"""
        for img_path in images:
            try:
                start_predict(
                    model_path,
                    img_path,
                    self.config.dataset_name,
                    model_name=Path(model_name).stem
                )
            except Exception as e:
                logger.error(f"预测失败 {img_path}: {e}")


# ========== 流水线主类 ==========

class Pipeline:
    """训练预测流水线"""
    
    def __init__(self, project_dir=None):
        self.project_dir = project_dir or get_project_root()
        self.config = PipelineConfig(self.project_dir)
    
    def setup(self, **kwargs):
        """
        配置流水线

        参数:
            do_train: 是否训练
            do_predict: 是否预测
            dataset_name: 数据集名称
            exp_name: 实验名称
            epochs: 训练轮数
            prediction_images: 预测图片路径（支持文件/文件夹混合）
            predict_models: 预测用的模型路径列表（None则使用刚训练/注册表的）
            model_names: 训练模型列表（如 ["yolov8s-obb.pt"] 或 ["/path/to/best.pt"]）
        """
        # 更新配置
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.model_names = kwargs.get('model_names', [])
        
        # 验证
        if self.config.do_train:
            if not self.config.dataset_name or not self.config.exp_name:
                raise ValueError("训练需要指定 dataset_name 和 exp_name")
        
        if self.config.do_predict:
            if not self.config.prediction_images:
                raise ValueError("预测需要指定 prediction_images")
    
    def run(self):
        """执行流水线"""
        logger.info(f"{'='*60}")
        logger.info("训练预测流水线")
        logger.info(f"  训练: {self.config.do_train}")
        logger.info(f"  预测: {self.config.do_predict}")
        logger.info(f"{'='*60}")
        
        trained_models = None
        
        # 训练阶段
        if self.config.do_train:
            trainer = Trainer(self.config)
            trained_models = trainer.train(self.model_names)
            
            if trained_models:
                save_model_registry(self.project_dir, trained_models)
        
        # 预测阶段
        if self.config.do_predict:
            predictor = Predictor(self.config)
            predict_models = predictor.get_predict_models(trained_models)
            
            if predict_models:
                predictor.predict(predict_models)
            else:
                logger.error("没有可用的模型进行预测")
        
        logger.info(f"{'='*60}")
        logger.info("流水线执行完成")
        logger.info(f"{'='*60}")


# ========== 使用示例 ==========

def main():
    """使用示例"""
    pipeline = Pipeline()
    project_dir = pipeline.project_dir
    
    # ===== 场景1: 完整流程（训练 + 预测）=====
    # 使用预训练模型 yolov8s-obb.pt 训练，然后用训练好的模型预测
    pipeline.setup(
        do_train=True,
        do_predict=True,
        dataset_name="booth_seg",
        exp_name="exp_v1",
        epochs=3,
        prediction_images=[f"{project_dir}/images/11届猪业.jpeg"],
        model_names=["yolov8s-obb.pt"]  # 使用默认预训练模型
    )
    # pipeline.run()
    
    # ===== 场景2: 只训练（不预测）=====
    # pipeline.setup(
    #     do_train=True,
    #     do_predict=False,
    #     dataset_name="booth_seg",
    #     exp_name="exp_v1",
    #     epochs=100,
    #     model_names=["yolov8s-obb.pt"]
    # )
    # pipeline.run()
    
    # ===== 场景3: 只预测（使用上次训练的模型）=====
    # pipeline.setup(
    #     do_train=False,
    #     do_predict=True,
    #     dataset_name="booth_seg",  # 用于输出目录命名
    #     prediction_images=[f"{project_dir}/images/"]
    #     # 不指定 predict_models，自动使用注册表中的模型
    # )
    # pipeline.run()
    
    # ===== 场景4: 基于已训练模型继续训练（增量训练）=====
    # pipeline.setup(
    #     do_train=True,
    #     do_predict=True,
    #     dataset_name="booth_seg",
    #     exp_name="exp_v2",
    #     epochs=50,
    #     prediction_images=[f"{project_dir}/images/"],
    #     model_names=[  # 指定已训练好的模型路径
    #         f"{project_dir}/output/models/yolov8s-obb/exp_v1/weights/best.pt"
    #     ]
    # )
    # pipeline.run()
    
    # ===== 场景5: 用任意模型预测（不训练）=====
    # pipeline.setup(
    #     do_train=False,
    #     do_predict=True,
    #     dataset_name="booth_seg",
    #     prediction_images=[f"{project_dir}/images/test.jpg"],
    #     predict_models=[  # 指定任意模型路径
    #         f"{project_dir}/my_custom_model.pt"
    #     ]
    # )
    # pipeline.run()
    
    pipeline.run()


if __name__ == "__main__":
    main()
