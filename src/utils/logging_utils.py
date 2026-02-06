"""
日志工具模块
提供统一的日志配置和格式化
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from src.core.constants import LOG_CONSTANTS


class ColoredFormatter(logging.Formatter):
    """彩色控制台日志格式化器"""

    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON 格式日志格式化器（便于日志分析）"""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import time

        log_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
            "timestamp_ms": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    获取配置好的 logger

    Args:
        name: logger 名称（通常使用脚本文件名）
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径（可选，默认输出到 logs/app.log）
        log_format: 日志格式 (colored, plain, json)

    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)

    # 防止重复添加 handler
    if logger.handlers:
        return logger

    # 设置日志级别
    if level is None:
        level = LOG_CONSTANTS.DEFAULT_LOG_LEVEL

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # 如果没有指定日志文件，使用默认的 logs/app.log
    if log_file is None:
        from src.utils.path_utils import get_project_root
        project_root = get_project_root()
        log_file = project_root / "logs" / "app.log"

    # 选择格式化器
    if log_format is None:
        log_format = LOG_CONSTANTS.LOG_FORMAT_COLOR

    if log_format == "json":
        formatter = JsonFormatter()
        console_format = "%(message)s"
    elif log_format == "plain":
        formatter = logging.Formatter(
            '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
        )
        console_format = None
    else:  # colored
        formatter = ColoredFormatter(
            '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_format = None

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    if console_format:
        console_handler.setFormatter(logging.Formatter(console_format))
    else:
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（始终启用，输出到统一文件）
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)

    return logger


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_format: str = "colored"
) -> None:
    """
    设置全局日志配置

    Args:
        level: 日志级别
        log_dir: 日志目录
        log_format: 日志格式
    """
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"
    else:
        log_file = None

    # 设置 root logger
    root_logger = get_logger("root", level=level, log_file=log_file, log_format=log_format)
    root_logger.info(f"Logging initialized with level: {level}")
