from ultralytics import YOLO
import os

# 1. 定义模型保存路径
# 请将 '/workspace/my_booth_project' 替换为你希望保存所有训练产出的实际目录

# 定义一个基础目录
project_dir = '/home/aistudio/YOLO/'
train_output_path = project_dir + 'models/train/'
# 为本次训练指定一个名称，训练文件会保存在以该名称命名的子文件夹中
exp_name = 'booth_seg_v1'

yolo_model = project_dir + "models/yolo11/yolo11m-seg.pt"
# yolo_model="/workspace/models/yolo26/yolo26m-seg.pt"

# 确保目录存在
os.makedirs(train_output_path, exist_ok=True)

# 2. 加载模型
# 注意：你原代码加载的是yolo11n.pt，这里已修正
model = YOLO(yolo_model)  # load a pretrained model (recommended for training)

# 3. 开始训练
results = model.train(
    # data='coco8-seg.yaml',  # 数据集配置文件路径
    data= project_dir + 'script/booth_seg.yaml',  # 数据集配置文件路径
    epochs=100,                               # 训练轮数
    imgsz=640,                                # 输入图像尺寸
    batch=-1,                                  # 批次大小，即每次训练迭代使用的样本数；-1： 自动计算最大可用 batch
    # augment=True,                             # 启用数据增强，以提高模型的泛化能力
    device=0,                                 # 训练设备
    # ================= 以下是新增/修改的关键参数 =================
    project=train_output_path,                      # 【新增】指定项目根目录
    name=exp_name,                            # 【新增】指定实验名称
    save=True,                                # 保存训练结果和模型（默认即为True）
    save_period=-1,                           # 每N轮保存一次检查点，-1表示仅在最后保存
    pretrained=True,                          # 从预训练权重开始（默认即为True）
    # ===========================================================
    workers=8,                                 # 工作线程数，用于数据加载和预处理
    amp=True,                                 # 开启混合精度训练
    # ================= 增加以下数据增强参数 =================
    degrees=180.0,    # 图像随机旋转角度范围 (-180, +180)
    flipud=0.5,       # 有 50% 的概率进行上下翻转
    fliplr=0.5,       # 有 50% 的概率进行左右翻转
    mosaic=1.0,       # 强烈建议开启：把四张图拼在一起，增加模型对小目标的感知
    close_mosaic=10,                            # 最后10轮关闭马赛克，有助于模型收敛（默认就是10）
    mixup=0.1,        # 进阶增强：图像融合，提高泛化能力
)

