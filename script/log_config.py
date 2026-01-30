import logging
import sys
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    """自定义日志格式化器，为不同级别的日志添加颜色"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'  # 重置颜色

    def format(self, record):
        # 如果是终端，则添加颜色
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                # 添加颜色到日志级别名称
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
                
                # 对整个日志消息应用颜色
                original_msg = record.msg
                record.msg = f"{self.COLORS[levelname]}{original_msg}{self.RESET}"
                
                # 获取格式化后的字符串
                formatted = super().format(record)
                
                # 恢复原始消息以便其他处理器可以使用
                record.msg = original_msg
                record.levelname = levelname
                
                return formatted
        # 如果不是终端，返回普通格式
        return super().format(record)


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
    
    # 创建彩色格式化器用于控制台
    colored_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器（使用彩色格式）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件，使用普通格式）
    if log_file:
        # 确保目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 文件日志不需要颜色
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
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