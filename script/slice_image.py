import cv2
import os
from pathlib import Path

# 导入工程化工具
from src.utils import get_logger, safe_mkdir

logger = get_logger('slice_image')


def slice_image(image_path, output_dir, tile_size=1024, overlap_ratio=0.2):
    """
    将大图切割成有重叠的小图块。

    参数:
        image_path: 输入大图的路径。
        output_dir: 输出小图块的文件夹路径。
        tile_size: 图块的宽度和高度（正方形），默认1024。
        overlap_ratio: 重叠率（0到1之间），默认0.2（20%）。
    """
    # 1. 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"错误：无法读取图片 {image_path}")
        return
    height, width = img.shape[:2]

    # 2. 创建输出目录
    safe_mkdir(output_dir)

    # 3. 计算步长（移动距离）
    step = int(tile_size * (1 - overlap_ratio))

    # 4. 计算图块数量（用于生成编号）
    count = 0
    for y in range(0, height - tile_size + 1, step):
        for x in range(0, width - tile_size + 1, step):
            # 计算当前图块的坐标（防止越界）
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)

            # 如果剩下的部分不足以切出一个完整图块，则从末尾反推坐标
            if (x_end - x) < tile_size or (y_end - y) < tile_size:
                x = max(0, width - tile_size)
                y = max(0, height - tile_size)
                x_end = x + tile_size
                y_end = y + tile_size

            # 5. 截取图块
            tile = img[y:y_end, x:x_end]

            # 6. 生成唯一文件名并保存
            # 文件名格式：原图名_行号_列号_块大小.jpg
            filename = f"{Path(image_path).stem}_{count:04d}_{x}_{y}_{tile_size}.jpg"
            output_path = Path(output_dir) / filename
            cv2.imwrite(str(output_path), tile)
            count += 1

    logger.info(f"切割完成！共生成 {count} 个图块，保存在 {output_dir}")


if __name__ == "__main__":
    from src.utils import get_project_root

    project_root = get_project_root()
    input_image_path = project_root / 'images' / '2024年展位图.jpg'
    output_directory = project_root / 'output' / 'slice_images'

    # 调用函数（参数采用推荐值）
    slice_image(input_image_path, output_directory, tile_size=1024, overlap_ratio=0.2)
