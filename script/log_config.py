import logging
import sys
from pathlib import Path

def setup_logger(name=__name__, level=logging.INFO, log_file=None):
    """
    设置项目通用日志记录器
    
    Args:
        name: logger的名称，默认使用模块名
        level: 日志级别，默认INFO
        log_file: 可选的日志文件路径，如果提供则同时输出到文件
    
    Returns:
        配置好的logger实例
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        # 确保目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name=__name__, log_file=None):
    """
    获取一个预配置的logger实例
    
    Args:
        name: logger的名称，默认使用模块名
        log_file: 可选的日志文件路径
    
    Returns:
        配置好的logger实例
    """
    return setup_logger(name, logging.INFO, log_file)


# 预定义的常用logger
def get_project_logger(script_name, log_dir="logs"):
    """
    为项目脚本获取专用的logger
    
    Args:
        script_name: 脚本名称，用于标识日志来源
        log_dir: 日志文件存储目录
    
    Returns:
        适合项目使用的logger实例
    """
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    log_file_path = project_root / log_dir / f"{script_name}.log"
    
    return get_logger(f"YOLO.{script_name}", log_file_path)