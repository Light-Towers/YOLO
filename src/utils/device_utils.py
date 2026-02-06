"""
设备工具模块
统一处理计算设备（CPU/GPU）相关逻辑
"""
import torch
from typing import Union, Optional


def get_device(device: Optional[Union[str, int]] = None) -> str:
    """
    获取计算设备

    Args:
        device: 设备指定
            - None: 自动选择（优先CUDA）
            - "cpu": 强制使用CPU
            - 0, 1, ...: 使用指定GPU编号
            - "cuda:0", "cuda:1", ...: 使用指定CUDA设备

    Returns:
        设备字符串，如 "cpu" 或 "cuda:0"
    """
    if device is not None:
        if isinstance(device, int):
            if torch.cuda.is_available() and device < torch.cuda.device_count():
                return f"cuda:{device}"
            return "cpu"
        if isinstance(device, str):
            if device.lower() == "cpu":
                return "cpu"
            if device.lower() == "auto":
                if torch.cuda.is_available():
                    return "cuda:0"
                return "cpu"
            # 验证CUDA设备是否可用
            if device.startswith("cuda:"):
                device_num = int(device.split(":")[1])
                if torch.cuda.is_available() and device_num < torch.cuda.device_count():
                    return device
            return device

    # 自动选择
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def get_available_gpus() -> int:
    """获取可用的GPU数量"""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_gpu_info(device: str = "cuda:0") -> dict:
    """获取GPU信息"""
    if not torch.cuda.is_available():
        return {"available": False}

    try:
        device_num = int(device.split(":")[1])
        props = torch.cuda.get_device_properties(device_num)
        return {
            "available": True,
            "name": props.name,
            "total_memory": props.total_memory,
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
    except Exception:
        return {"available": True, "error": "Failed to get GPU info"}


def clear_cuda_cache():
    """清理CUDA缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
