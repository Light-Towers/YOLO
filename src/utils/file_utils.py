"""
文件工具模块
提供统一的文件读写操作
"""
import json
from pathlib import Path
from typing import Any, Union

import yaml

PathLike = Union[str, Path]


def read_yaml(yaml_path: PathLike) -> dict:
    """
    读取 YAML 文件

    Args:
        yaml_path: YAML 文件路径

    Returns:
        解析后的字典

    Raises:
        FileNotFoundError: 文件不存在
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML 文件不存在: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_yaml(yaml_path: PathLike, data: dict) -> None:
    """
    写入 YAML 文件

    Args:
        yaml_path: YAML 文件路径
        data: 要写入的数据
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def read_json(json_path: PathLike) -> Any:
    """
    读取 JSON 文件

    Args:
        json_path: JSON 文件路径

    Returns:
        解析后的数据

    Raises:
        FileNotFoundError: 文件不存在
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(json_path: PathLike, data: Any, indent: int = 4) -> None:
    """
    写入 JSON 文件

    Args:
        json_path: JSON 文件路径
        data: 要写入的数据
        indent: 缩进空格数
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_text(text_path: PathLike, encoding: str = 'utf-8') -> str:
    """
    读取文本文件

    Args:
        text_path: 文本文件路径
        encoding: 文件编码

    Returns:
        文件内容
    """
    text_path = Path(text_path)
    with open(text_path, 'r', encoding=encoding) as f:
        return f.read()


def write_text(text_path: PathLike, content: str, encoding: str = 'utf-8') -> None:
    """
    写入文本文件

    Args:
        text_path: 文本文件路径
        content: 文件内容
        encoding: 文件编码
    """
    text_path = Path(text_path)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    with open(text_path, 'w', encoding=encoding) as f:
        f.write(content)
