"""图像切片工具模块"""

from typing import List, Optional, Tuple


class TileCalculator:
    """图像切片计算器"""

    @staticmethod
    def calculate_tiles(
        image_size: Tuple[int, int],
        tile_size: int,
        overlap: Optional[int] = None,
        overlap_ratio: Optional[float] = None,
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
        if overlap is not None and 0 < overlap < 1:
            overlap_ratio = overlap
            overlap = None

        # 计算步长
        if overlap is not None:
            step = tile_size - overlap
        elif overlap_ratio is not None:
            step = int(tile_size * (1 - overlap_ratio))
        else:
            step = tile_size

        h, w = image_size

        # 新算法：从图像边界开始，固定步长切分，确保覆盖整个图像且无重复
        tiles = []
        tile_id = 0

        # 计算需要多少行/列才能覆盖整个图像
        num_cols = (w - tile_size + step - 1) // step + 1 if w > tile_size else 1
        num_rows = (h - tile_size + step - 1) // step + 1 if h > tile_size else 1

        # 调试输出
        print(f'    图像尺寸: {w}x{h}')
        print(f'    tile_size: {tile_size}')
        print(f'    step: {step}')
        print(f'    网格: {num_cols}列 x {num_rows}行')

        for row in range(num_rows):
            # 计算y起始位置：正常步长，最后一行调整为确保tile在图像内
            if row == num_rows - 1 and num_rows > 1:
                y = max(0, h - tile_size)
            else:
                y = row * step

            for col in range(num_cols):
                # 计算x起始位置：正常步长，最后一列调整为确保tile在图像内
                if col == num_cols - 1 and num_cols > 1:
                    x = max(0, w - tile_size)
                else:
                    x = col * step

                # 确保不超出图像边界
                x = max(0, min(x, w - tile_size))
                y = max(0, min(y, h - tile_size))

                # 固定大小的切片
                x_end = x + tile_size
                y_end = y + tile_size

                tiles.append((tile_id, x, y, x_end, y_end))
                tile_id += 1

        print(f'    实际生成切片数: {len(tiles)}')

        return tiles

    @staticmethod
    def get_tile_coordinates(
        image_size: Tuple[int, int],
        tile_size: int,
        overlap: Optional[int] = None,
        overlap_ratio: Optional[float] = None,
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
                "tile_id": tile_id,
                "x": x,
                "y": y,
                "width": x_end - x,
                "height": y_end - y,
            }
            for tile_id, x, y, x_end, y_end in tiles
        ]


def calculate_tiles(
    image_size: Tuple[int, int],
    tile_size: int,
    overlap: Optional[int] = None,
    overlap_ratio: Optional[float] = None,
) -> List[Tuple[int, int, int, int, int]]:
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
