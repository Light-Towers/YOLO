# show_label.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import numpy as np

# 设置PIL最大图像尺寸限制，以处理大图像
Image.MAX_IMAGE_PIXELS = None  # 关闭像素限制检查，或者可以设为一个更大的值

# 1. 加载展位图文件
img_path = 'd:/0-mingyang/img_handle/场馆展位图/第十一届世界猪业博览会.jpeg'
img = Image.open(img_path)

# 图像缩放处理，如果图像过大则进行缩放
max_width, max_height = 3000, 3000  # 设置最大尺寸
original_width, original_height = img.size
scale_factor = min(max_width/original_width, max_height/original_height)

if scale_factor < 1:
    new_size = (int(original_width * scale_factor), int(original_height * scale_factor))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    print(f"图像已缩放，缩放因子: {scale_factor:.3f}")

fig, ax = plt.subplots(figsize=(12, 16))

# 显示原图
ax.imshow(img)

# 示例数据，实际使用时请从JSON文件加载
data = [
    {
        "name": "item",
        "class": 0,
        "confidence": 0.9631933569908142,
        "poly": [
            748.6508178710938,
            564.3227691650391,
            749.681640625,
            473.8530731201172,
            583.4720458984375,
            471.95921325683594,
            582.4412231445312,
            562.4289093017578
        ]
    }
]


# 2. 从JSON文件加载数据
# 请将 'your_data.json' 替换为实际的JSON文件路径
# json_file_path = 'c:/Users/osmondy/Downloads/obb_result_2024年展位图_v12.json'  # 可以替换为实际的JSON文件路径
# with open(json_file_path, 'r', encoding='utf-8') as f:
#     data = json.load(f)


# 3. 循环绘制每个检测到的物体
for obj in data:
    # 提取 OBB poly 数据: [x1, y1, x2, y2, x3, y3, x4, y4] 四个角点坐标
    poly = obj["poly"]
    
    # 如果进行了缩放，需要按比例调整坐标
    if scale_factor < 1:
        poly = [coord * scale_factor for coord in poly]
    
    # 将8个坐标转换为x,y坐标对
    x_coords = [poly[i] for i in range(0, 8, 2)]  # 奇数位是x坐标
    y_coords = [poly[i] for i in range(1, 8, 2)]  # 偶数位是y坐标
    
    # 绘制四边形
    polygon = patches.Polygon(
        list(zip(x_coords, y_coords)), 
        linewidth=1, 
        edgecolor='red', 
        facecolor='none'
    )
    ax.add_patch(polygon)
    
    # # 计算中心点作为标签位置
    # center_x = sum(x_coords) / len(x_coords)
    # center_y = sum(y_coords) / len(y_coords)
    # 
    # # 准备显示的标签内容：名称 + 置信度
    # label_text = f"{obj['name']} {obj['confidence']:.2f}"
    # 
    # # 在四边形中心标注名称
    # ax.text(center_x, center_y, label_text, 
    #         color='white', fontweight='bold', fontsize=10,
    #         ha='center', va='center',
    #         bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))

# 4. 保存图片而不是显示
output_path = 'd:/0-mingyang/img_handle/场馆展位图/2024年展位图_with_annotations.jpg'  # 可以根据需要调整输出路径
plt.title("Object Detection Visualization")
plt.axis('on')
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存为图片，dpi=300表示高分辨率
print(f"图片已保存至: {output_path}")