"""图像切片工具模块"""
import numpy as np
from typing import List, Tuple, Optional


class TileCalculator:
    """图像切片计算器"""
    
    @staticmethod
    def calculate_tiles(
        image_size: Tuple[int, int],
        tile_size: int,
        overlap: Optional[int] = None,
        overlap_ratio: Optional[float] = None
    ) -> List[Tuple[int, int, int, int, int]]:
        """
        计算图像的所有切片位置
        
        Args:
            image_size: 图像尺寸 (height, width)
            tile_size: 切片大小（正方形边长）
            overlap: 重叠像素值（与overlap_ratio二选一）
            overlap_ratio: 重叠比例 0-1（与overlap二选一）
            
        Returns:
            切片列表，每个元素为 (tile_id, x, y, x_end, y_end)
        """
        # 处理重叠参数
        if overlap_ratio is not None:
            overlap = int(tile_size * overlap_ratio)
        elif overlap is None:
            overlap = 0
        
        step = tile_size - overlap
        h, w = image_size
        
        tiles = []
        tile_id = 0
        y = 0
        
        while y < h:
            x = 0
            while x < w:
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                tiles.append((tile_id, x, y, x_end, y_end))
                tile_id += 1
                x += step
            y += step
        
        return tiles
    
    @staticmethod
    def get_tile_coordinates(
        image_size: Tuple[int, int],
        tile_size: int,
        overlap: Optional[int] = None,
        overlap_ratio: Optional[float] = None
    ) -> List[dict]:
        """
        获取切片坐标信息（返回字典格式）
        
        Args:
            image_size: 图像尺寸 (height, width)
            tile_size: 切片大小
            overlap: 重叠像素值
            overlap_ratio: 重叠比例
            
        Returns:
            切片信息列表，每个元素包含 x, y, width, height
        """
        tiles = TileCalculator.calculate_tiles(image_size, tile_size, overlap, overlap_ratio)
        return [
            {
                'tile_id': tile_id,
                'x': x,
                'y': y,
                'width': x_end - x,
                'height': y_end - y
            }
            for tile_id, x, y, x_end, y_end in tiles
        ]


def calculate_tiles(image_size: Tuple[int, int], tile_size: int,
                   overlap: Optional[int] = None, overlap_ratio: Optional[float] = None) -> List[Tuple[int, int, int, int, int]]:
    """
    便捷函数：计算图像的所有切片位置
    
    Args:
        image_size: 图像尺寸 (height, width)
        tile_size: 切片大小（正方形边长）
        overlap: 重叠像素值
        overlap_ratio: 重叠比例 0-1
        
    Returns:
        切片列表，每个元素为 (tile_id, x, y, x_end, y_end)
    """
    return TileCalculator.calculate_tiles(image_size, tile_size, overlap, overlap_ratio)
