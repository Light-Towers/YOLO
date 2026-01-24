from ultralytics import YOLO
import os

# 1. 定义模型保存路径
# 请将 '/home/aistudio/YOLO/' 替换为你希望保存所有训练产出的实际目录

# 定义一个基础目录
project_dir = '/home/aistudio/YOLO/'

# 指定你希望统一存放基础模型（如 yolo26n.pt）的目录
os.environ['YOLO_ASSETS'] = project_dir + 'models/'

train_output_path = project_dir + 'models/train/'
# 为本次训练指定一个名称，训练文件会保存在以该名称命名的子文件夹中
exp_name = 'booth_obb_v1'  # 修改实验名称以区分OBB任务

# ========== 关键修改1：使用OBB模型 ==========
# 选择适合的OBB模型，注意模型后缀是-obb
yolo_model = project_dir + "models/yolo11/yolo11m-obb.pt"  # 推荐使用yolo11m-obb.pt
# 或使用YOLOv8的OBB模型（如果你有的话）
# yolo_model = project_dir + "models/yolov8/yolov8m-obb.pt"

# 确保目录存在
os.makedirs(train_output_path, exist_ok=True)

# 2. 加载模型
model = YOLO(yolo_model)  # 加载一个预训练的OBB模型

# 3. 开始训练
results = model.train(
    # ========== 关键修改2：指定任务类型 ==========
    task='obb',  # 明确指定任务类型为旋转目标检测
    
    # 数据集配置文件路径（可以使用同一个yaml文件，但确保标签格式正确）
    data= project_dir + 'script/booth_seg.yaml',  # 确保这个yaml文件指向OBB格式的标签
    
    epochs=300,                               # 训练轮数
    patience=50,                              # 早停耐心值
    imgsz=640,                                # 输入图像尺寸
    batch=-1,                                 # 自动计算最大可用batch
    device=0,                                 # 训练设备
    
    # ========== 项目相关参数 ==========
    project=train_output_path,                # 指定项目根目录
    name=exp_name,                            # 指定实验名称
    save=True,                                # 保存训练结果和模型
    save_period=-1,                           # 仅在最后保存检查点
    pretrained=True,                          # 从预训练权重开始
    
    # ========== 训练优化参数 ==========
    workers=8,                                # 工作线程数
    amp=True,                                 # 开启混合精度训练
    
    # ========== 关键修改3：调整数据增强策略 ==========
    # OBB任务对旋转敏感，需要谨慎调整旋转增强
    degrees=15.0,      # 【建议调低】展位图通常视角固定，避免过大的旋转
    translate=0.1,     # 平移增强
    scale=0.5,         # 缩放增强
    shear=0.0,         # 【建议关闭】剪切变换可能破坏旋转框的角度信息
    perspective=0.001, # 透视变换，保持较小的值
    flipud=0.0,        # 上下翻转【建议关闭】
    fliplr=0.5,        # 左右翻转可保留
    
    # 马赛克增强相关
    mosaic=1.0,        # 开启马赛克增强
    mixup=0.1,         # MixUp增强，不宜过高
    copy_paste=0.0,    # 【建议关闭】复制粘贴增强可能不适合OBB
    
    # ========== 关键修改4：OBB特定参数 ==========
    # YOLO OBB任务会自动处理旋转框，以下是可能需要关注的参数
    overlap_mask=False,  # 【注意】OBB任务不需要掩码重叠，应该设为False
    single_cls=True,     # 如果你的数据集中只有"展位"一个类别，设为True
    
    # ========== 优化器与学习率 ==========
    optimizer='auto',    # 自动选择优化器
    lr0=0.01,           # 初始学习率
    lrf=0.01,           # 最终学习率系数 (lr0 * lrf)
    momentum=0.937,     # 动量
    weight_decay=0.0005, # 权重衰减
    warmup_epochs=3,    # 学习率预热轮数
    warmup_momentum=0.8, # 预热期动量
    warmup_bias_lr=0.1, # 预热期偏置学习率
    
    # ========== 其他调整 ==========
    dropout=0.0,        # OBB任务通常不需要dropout
    cos_lr=True,        # 使用余弦退火学习率调度
    label_smoothing=0.0, # 标签平滑
    
    # ========== 验证相关参数 ==========
    val=True,           # 在训练期间进行验证
    plots=True,         # 在训练期间生成并保存图表
    exist_ok=True,      # 如果实验目录已存在，则覆盖
    resume=False,       # 是否从最近的检查点恢复训练
    
    # ========== 针对密集小目标的调整 ==========
    # 如果你的展位密集且较小，可以考虑以下调整
    # multi_scale=False,  # 多尺度训练（会增加训练时间）
    # nbs=64,             # 名义批量大小
    
    # ========== 调试参数 ==========
    verbose=True,       # 输出详细信息
    deterministic=True, # 确保可重复性
)

print("训练完成！")