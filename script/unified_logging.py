#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO脚本统一日志配置

使用方法：
在调用任何YOLO脚本之前，导入此模块并调用 set_unified_log_file() 函数
所有脚本日志将输出到同一个文件中

示例：
    from log_config import set_unified_log_file
    from pathlib import Path

    # 设置统一日志文件路径
    project_root = Path(__file__).resolve().parent
    log_file = project_root / 'logs' / 'unified_yolo.log'
    set_unified_log_file(log_file)

    # 然后执行其他脚本...
    from train import main
    main()
"""

from pathlib import Path
from log_config import set_unified_log_file
from datetime import datetime

def setup_unified_logging(project_root=None, log_filename="unified_yolo.log"):
    """
    设置统一日志配置

    Args:
        project_root: 项目根目录，默认为当前文件所在目录
        log_filename: 统一日志文件名，默认为 'unified_yolo.log'

    Returns:
        日志文件路径

    示例:
        # 方式1: 使用默认配置
        setup_unified_logging()

        # 方式2: 指定项目根目录和日志文件名
        setup_unified_logging(
            project_root=Path('/workspace/YOLO'),
            log_filename='my_training.log'
        )
    """
    if project_root is None:
        # 默认使用当前文件所在目录的上层目录
        project_root = Path(__file__).resolve().parent.parent

    # 创建logs目录
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志文件路径
    log_file_path = logs_dir / log_filename

    # 调用 log_config 的设置函数
    set_unified_log_file(log_file_path)

    # 添加时间戳记录
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    from log_config import logger
    logger.info("="*80)
    logger.info(f"统一日志配置已启用 - {timestamp}")
    logger.info(f"日志文件: {log_file_path}")
    logger.info("="*80)

    return log_file_path


if __name__ == "__main__":
    # 测试统一日志配置
    log_file = setup_unified_logging()
    print(f"统一日志已设置，所有日志将输出到: {log_file}")

    # 导入并测试其他脚本的logger
    from log_config import get_project_logger
    test_logger1 = get_project_logger('test_script1')
    test_logger1.info("这是来自 test_script1 的日志")

    test_logger2 = get_project_logger('test_script2')
    test_logger2.info("这是来自 test_script2 的日志")

    test_logger3 = get_project_logger('test_script3')
    test_logger3.warning("这是来自 test_script3 的警告信息")

    print("\n测试完成！请查看日志文件确认所有日志已合并。")
