import cv2
import os
import numpy as np

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
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    height, width = img.shape[:2]

    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

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
            filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{count:04d}_{x}_{y}_{tile_size}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, tile)
            count += 1

    print(f"切割完成！共生成 {count} 个图块，保存在 {output_dir}")

# ==== 使用示例 ====
# 请修改以下路径和参数
input_image_path = "D:/Study/github/YOLO/images/2024年展位图.jpg"  # 替换为你的图片实际路径
output_directory = "D:/0-mingyang/img_handle/YOLO_Train/slice_out_1024"                      # 替换为你希望的输出文件夹

# 调用函数（参数采用上方推荐值）
slice_image(input_image_path, output_directory, tile_size=1024, overlap_ratio=0.2)