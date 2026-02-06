"""
项目自定义异常类
定义统一的异常层次结构
"""


class YOLOProjectError(Exception):
    """项目基础异常类"""

    def __init__(self, message: str, *args, **kwargs):
        self.message = message
        super().__init__(message, *args, **kwargs)


class DatasetError(YOLOProjectError):
    """数据集相关错误"""
    pass


class ModelNotFoundError(YOLOProjectError):
    """模型文件未找到错误"""

    def __init__(self, model_path: str, suggestion: str = ""):
        message = f"模型文件不存在: {model_path}"
        if suggestion:
            message += f"\n建议: {suggestion}"
        super().__init__(message)


class ModelLoadError(YOLOProjectError):
    """模型加载失败错误"""
    pass


class ConfigurationError(YOLOProjectError):
    """配置错误"""
    pass


class InvalidAnnotationError(DatasetError):
    """无效标注错误"""
    pass


class ImageProcessingError(YOLOProjectError):
    """图像处理错误"""
    pass


class InferenceError(YOLOProjectError):
    """推理错误"""
    pass


class TrainingError(YOLOProjectError):
    """训练错误"""
    pass


class PathError(YOLOProjectError):
    """路径错误"""
    pass


class DatasetNotFoundError(DatasetError):
    """数据集未找到错误"""

    def __init__(self, dataset_name: str, search_path: str):
        message = f"数据集 '{dataset_name}' 未找到"
        message += f"\n搜索路径: {search_path}"
        super().__init__(message)


class DatasetConfigError(DatasetError):
    """数据集配置错误"""

    def __init__(self, config_path: str, reason: str):
        message = f"数据集配置错误: {config_path}"
        message += f"\n原因: {reason}"
        super().__init__(message)
