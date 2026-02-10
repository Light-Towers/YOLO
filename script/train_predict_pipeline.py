from ultralytics import YOLO
import os
from pathlib import Path
import re
import torch

# 导入工程化工具
from src.utils import (
    get_project_root,
    get_logger,
    get_image_files,
)
from src.core import TRAINING_CONSTANTS

# 导入train模块中的函数
from train import train_model, get_model_path, update_dataset_path

# 导入predict_sahi模块
from predict_sahi import start_predict

# 获取项目logger
logger = get_logger('train_predict_pipeline')


class TrainingPipelineConfig:
    """训练预测流水线配置类"""
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.models_to_train = []
        self.dataset_name = None
        self.exp_name = None
        self.prediction_images = []
        self.epochs = 3

    def set_models(self, models):
        """设置待训练模型列表"""
        self.models_to_train = models

    def set_dataset(self, dataset_name):
        """设置数据集名称"""
        self.dataset_name = dataset_name

    def set_experiment(self, exp_name):
        """设置实验名称"""
        self.exp_name = exp_name

    def set_prediction_images(self, images):
        """设置预测图像路径列表"""
        self.prediction_images = images

    def set_epochs(self, epochs):
        """设置训练轮数"""
        self.epochs = epochs

    def validate(self):
        """验证配置是否完整"""
        if not self.models_to_train:
            raise ValueError("未设置待训练模型列表")
        if not self.dataset_name:
            raise ValueError("未设置数据集名称")
        if not self.exp_name:
            raise ValueError("未设置实验名称")
        if not self.prediction_images:
            raise ValueError("未设置预测图像路径")


class DatasetManager:
    """数据集管理类"""
    def __init__(self, project_dir, dataset_name, logger):
        self.project_dir = project_dir
        self.dataset_name = dataset_name
        self.dataset_root = project_dir / 'datasets' / dataset_name
        self.dataset_yaml_path = self.dataset_root / 'dataset.yaml'
        self.logger = logger

    def prepare(self):
        """准备数据集（更新路径等）"""
        self.logger.info(f"Preparing dataset: {self.dataset_name}")
        self.logger.info(f"Dataset root: {self.dataset_root}")

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_root}")

        if not self.dataset_yaml_path.exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {self.dataset_yaml_path}")

        update_dataset_path(self.dataset_yaml_path, self.dataset_root)
        self.logger.info(f"Dataset {self.dataset_name} prepared successfully")

    def get_yaml_path(self):
        """获取数据集配置文件路径"""
        return self.dataset_yaml_path


class ModelTrainer:
    """模型训练管理类"""
    def __init__(self, config, logger):
        self.config = config
        self.project_dir = config.project_dir
        self.logger = logger
        self.trained_models = []

    def train_all_models(self):
        """训练所有配置的模型"""
        dataset_manager = DatasetManager(
            self.project_dir,
            self.config.dataset_name,
            self.logger
        )
        dataset_manager.prepare()

        self.logger.info(f"{'='*50}")
        self.logger.info(f"开始训练 {len(self.config.models_to_train)} 个模型")
        self.logger.info(f"{'='*50}")

        for model_filename in self.config.models_to_train:
            model_path = self._train_single_model(model_filename)
            if model_path:
                self.trained_models.append((model_path, model_filename))

        self.logger.info(f"{'='*50}")
        self.logger.info(f"训练完成，成功训练 {len(self.trained_models)} 个模型")
        self.logger.info(f"{'='*50}")

        return self.trained_models

    def _train_single_model(self, model_filename):
        """训练单个模型"""
        self.logger.info(f"{'='*50}")
        self.logger.info(f"开始训练模型: {model_filename}")
        self.logger.info(f"{'='*50}")

        yolo_model_path = get_model_path(model_filename, self.project_dir)
        if not yolo_model_path.exists():
            self.logger.warning(f"预训练模型未找到 {yolo_model_path}，将使用自动下载")

        dataset_manager = DatasetManager(
            self.project_dir,
            self.config.dataset_name,
            self.logger
        )

        best_model_path = train_model(
            yolo_model_path,
            dataset_manager.get_yaml_path(),
            self.project_dir,
            self.config.exp_name,
            self.config.dataset_name,
            epochs=self.config.epochs
        )

        if best_model_path:
            self.logger.info(f"模型 {model_filename} 训练完成")
        else:
            self.logger.warning(f"模型 {model_filename} 训练失败")

        return best_model_path


class PredictionManager:
    """预测管理类"""
    def __init__(self, config, logger):
        self.config = config
        self.project_dir = config.project_dir
        self.logger = logger

    @staticmethod
    def get_image_paths(image_sources, logger):
        """从图片路径列表或文件夹路径列表中获取所有图片路径"""
        image_paths = []
        for source in image_sources:
            source_path = Path(source)
            if source_path.is_file():
                image_paths.append(str(source_path))
            elif source_path.is_dir():
                # 使用工具函数获取图片文件
                images = get_image_files(source_path, recursive=False)
                image_paths.extend([str(img) for img in images])
            else:
                logger.warning(f"图像源不存在: {source}")

        return sorted(list(set(image_paths)))

    def run_predictions(self, trained_models):
        """对所有训练好的模型运行预测"""
        all_image_paths = self.get_image_paths(self.config.prediction_images, self.logger)
        self.logger.info(f"找到 {len(all_image_paths)} 张待预测图像")

        if not all_image_paths:
            self.logger.warning("没有找到待预测图像，跳过预测阶段")
            return

        self.logger.info(f"{'='*50}")
        self.logger.info("开始对所有模型进行预测")
        self.logger.info(f"{'='*50}")

        for model_path, model_filename in trained_models:
            self.logger.info(f"正在使用模型 {model_filename} 进行预测")
            self._predict_single_model(model_path, model_filename, all_image_paths)

    def _predict_single_model(self, model_path, model_filename, image_paths):
        """使用单个模型预测所有图像"""
        model_name_for_output = Path(model_filename).stem

        for image_path in image_paths:
            self.logger.info(f"正在预测图像: {image_path}")
            try:
                start_predict(
                    model_path,
                    image_path,
                    self.config.dataset_name,
                    model_name=model_name_for_output
                )
                self.logger.info(f"图像 {image_path} 预测完成")
            except Exception as e:
                self.logger.error(f"预测图像 {image_path} 时出错: {str(e)}")


class TrainPredictPipeline:
    """训练预测流水线主类"""
    def __init__(self, project_dir=None):
        self.project_dir = project_dir or get_project_root()
        self.logger = get_logger('train_predict_pipeline')
        self.logger.info(f"项目根目录: {self.project_dir}")

        self.config = TrainingPipelineConfig(self.project_dir)

    def load_config(self, models, dataset_name, exp_name, prediction_images, epochs=3):
        """加载配置"""
        self.config.set_models(models)
        self.config.set_dataset(dataset_name)
        self.config.set_experiment(exp_name)
        self.config.set_prediction_images(prediction_images)
        self.config.set_epochs(epochs)
        self.config.validate()

    def run(self):
        """执行完整的训练预测流水线"""
        self.logger.info(f"{'='*60}")
        self.logger.info("开始执行训练预测流水线")
        self.logger.info(f"{'='*60}")

        # 训练阶段
        trainer = ModelTrainer(self.config, self.logger)
        trained_models = trainer.train_all_models()

        if not trained_models:
            self.logger.error("没有成功训练的模型，终止流水线")
            return

        # 预测阶段
        predictor = PredictionManager(self.config, self.logger)
        predictor.run_predictions(trained_models)

        self.logger.info(f"{'='*60}")
        self.logger.info("所有训练任务和预测任务已完成！")
        self.logger.info(f"{'='*60}")


def main():
    """主函数：配置并运行训练预测流水线"""
    pipeline = TrainPredictPipeline()

    # 配置流水线参数
    models = [
        "yolov8s-obb.pt",
        "yolo11s-obb.pt",
        "yolo26s-obb.pt",
    ]

    dataset_name = "booth_seg"
    exp_name = 'booth_obb_v1'

    project_dir = pipeline.project_dir
    prediction_images = [
        # f"{project_dir}/images/2024年展位图.jpg",
        # f"{project_dir}/images/第十一届世界猪业博览会.jpeg",
        # f"{project_dir}/images/长沙国际会展中心.jpg",
        # f"{project_dir}/images/2020畜博会.png",

        f"{project_dir}/images/",
    ]

    # 加载配置并运行
    pipeline.load_config(
        models=models,
        dataset_name=dataset_name,
        exp_name=exp_name,
        prediction_images=prediction_images,
        epochs=300
    )
    pipeline.run()


if __name__ == "__main__":
    main()