import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

# 导入工程化工具
from src.utils import get_logger, read_json

logger = get_logger('label.show_label')


# 设置PIL最大图像尺寸限制，以处理大图像
Image.MAX_IMAGE_PIXELS = None


def show_label_with_annotations(image_path, json_path, output_path=None):
    """
    显示图像并在其上绘制标注

    参数:
        image_path: 输入图像路径
        json_path: 标注JSON文件路径（包含poly字段）
        output_path: 输出图像路径（可选），不提供则仅显示
    """
    # 1. 加载展位图文件
    img = Image.open(image_path)
    logger.info(f"加载图像: {image_path}")
    logger.info(f"图像原始尺寸: {img.size}")

    # 2. 图像缩放处理，如果图像过大则进行缩放
    max_width, max_height = 3000, 3000
    original_width, original_height = img.size
    scale_factor = min(max_width / original_width, max_height / original_height)

    if scale_factor < 1:
        new_size = (int(original_width * scale_factor), int(original_height * scale_factor))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"图像已缩放，缩放因子: {scale_factor:.3f}, 新尺寸: {new_size}")

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(img)

    # 3. 从JSON文件加载数据
    data = read_json(json_path)
    logger.info(f"加载标注: {json_path}, 共 {len(data)} 个检测对象")

    # 4. 循环绘制每个检测到的物体
    for i, obj in enumerate(data):
        # 提取 OBB poly 数据: [x1, y1, x2, y2, x3, y3, x4, y4] 四个角点坐标
        if "poly" not in obj:
            continue

        poly = obj["poly"]

        # 如果进行了缩放，需要按比例调整坐标
        if scale_factor < 1:
            poly = [coord * scale_factor for coord in poly]

        # 将8个坐标转换为x,y坐标对
        x_coords = [poly[i] for i in range(0, 8, 2)]
        y_coords = [poly[i] for i in range(1, 8, 2)]

        # 绘制四边形
        polygon = patches.Polygon(
            list(zip(x_coords, y_coords)),
            linewidth=1,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(polygon)

        # 计算中心点作为标签位置
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        # 准备显示的标签内容：名称 + 置信度
        label_text = f"{obj.get('name', 'item')} {obj.get('confidence', 0):.2f}"

        # 在四边形中心标注名称
        ax.text(center_x, center_y, label_text,
                color='white', fontweight='bold', fontsize=10,
                ha='center', va='center',
                bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))

    plt.title("Object Detection Visualization")
    plt.axis('on')

    # 5. 保存或显示图片
    if output_path:
        safe_path = Path(output_path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(safe_path), dpi=300, bbox_inches='tight')
        logger.info(f"图片已保存至: {output_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from src.utils import get_project_root

    project_root = get_project_root()

    # 使用示例
    image_path = project_root / 'images' / '2024年展位图.jpg'
    json_path = project_root / 'output' / 'results' / 'booth_obb_v13' / 'booth' / 'HongMu_sahi.json'
    output_path = project_root / 'output' / 'test' / '2024年展位图_with_annotations.jpg'

    show_label_with_annotations(image_path, json_path, output_path)
