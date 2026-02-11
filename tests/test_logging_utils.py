"""
测试日志工具模块
"""
import logging
from pathlib import Path

import pytest

from src.utils.logging_utils import (
    ColoredFormatter,
    JsonFormatter,
    get_logger,
    setup_logging,
)


class TestLoggingUtils:
    """日志工具测试"""

    def test_get_logger_basic(self):
        """测试获取基础 logger"""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_get_logger_with_level(self):
        """测试获取带级别的 logger"""
        logger = get_logger("test_logger", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_get_logger_with_colored_format(self):
        """测试获取彩色格式的 logger"""
        logger = get_logger("test_logger", log_format="colored")
        assert logger.handlers[0].formatter.__class__ == ColoredFormatter

    def test_get_logger_with_plain_format(self):
        """测试获取普通格式的 logger"""
        logger = get_logger("test_logger", log_format="plain")
        assert isinstance(logger.handlers[0].formatter, logging.Formatter)
        assert not isinstance(logger.handlers[0].formatter, ColoredFormatter)

    def test_get_logger_with_json_format(self):
        """测试获取 JSON 格式的 logger"""
        logger = get_logger("test_logger", log_format="json")
        assert isinstance(logger.handlers[0].formatter, JsonFormatter)

    def test_get_logger_with_file(self, tmp_path):
        """测试获取带文件输出的 logger"""
        log_file = tmp_path / "test.log"
        logger = get_logger("test_logger", log_file=log_file, level="INFO")

        logger.info("Test message")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_colored_formatter_format(self):
        """测试彩色格式化器"""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted

    def test_json_formatter_format(self):
        """测试 JSON 格式化器"""
        formatter = JsonFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        import json
        parsed = json.loads(formatted)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_setup_logging(self, tmp_path):
        """测试设置全局日志"""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir, level="INFO")

        root_logger = logging.getLogger("root")
        assert root_logger.level == logging.INFO
