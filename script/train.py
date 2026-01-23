from ultralytics import YOLO
import os

# 1. 定义模型保存路径
# 请将 '/workspace/my_booth_project' 替换为你希望保存所有训练产出的实际目录
project_dir = '/workspace/models/train'
# 为本次训练指定一个名称，训练文件会保存在以该名称命名的子文件夹中
exp_name = 'booth_detection_v1'

# 确保目录存在
os.makedirs(project_dir, exist_ok=True)

# 2. 加载模型
# 注意：你原代码加载的是yolo11n.pt，这里已修正
# 如果你想使用YOLOv8，请将参数改为 'yolov8n.pt'
model = YOLO('/workspace/models/yolo11/yolo11n.pt')

# 3. 开始训练
results = model.train(
    data='/workspace/script/data_test.yaml',  # 数据集配置文件路径
    epochs=100,                               # 训练轮数
    imgsz=416,                                # 输入图像尺寸
    batch=8,                                  # 批次大小，即每次训练迭代使用的样本数
    augment=True,                             # 启用数据增强，以提高模型的泛化能力
    device=0,                             # 训练设备
    # ================= 以下是新增/修改的关键参数 =================
    project=project_dir,                      # 【新增】指定项目根目录
    name=exp_name,                            # 【新增】指定实验名称
    save=True,                                # 保存训练结果和模型（默认即为True）
    save_period=-1,                           # 每N轮保存一次检查点，-1表示仅在最后保存
    pretrained=True,                          # 从预训练权重开始（默认即为True）
    # ===========================================================
)

print(f"训练完成！模型和结果已保存至: {os.path.join(project_dir, exp_name)}")