# YOLO 模型层级结构与 freeze 参数笔记

## 1. 模型基本信息

以 `yolov8s-obb` 为例：

| 指标 | 数值 |
|------|------|
| 官方统计层数 | 145 层 |
| 实际子层数 | 265 层 |
| 顶层模块数 | 23 个 (model.0 ~ model.22) |
| 参数总量 | 11,422,166 (~11.4M) |
| 计算量 | 29.6 GFLOPs |

---

## 2. 模型架构划分

### 2.1 划分依据

来源：模型配置文件 `ultralytics/cfg/models/v8/yolov8-obb.yaml`

```yaml
# Backbone 部分 (第17-29行)
backbone:
  - [-1, 1, Conv, [64, 3, 2]]    # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]   # 1-P2/4
  - [-1, 3, C2f, [128, True]]    # 2
  - [-1, 1, Conv, [256, 3, 2]]   # 3-P3/8
  - [-1, 6, C2f, [256, True]]    # 4
  - [-1, 1, Conv, [512, 3, 2]]   # 5-P4/16
  - [-1, 6, C2f, [512, True]]    # 6
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]   # 8
  - [-1, 1, SPPF, [1024, 5]]     # 9

# Head 部分 (第31-49行)，包含 Neck 和检测头
head:
  - [-1, 1, nn.Upsample, ...]    # 10-21: Neck
  - [[15, 18, 21], 1, OBB, [nc, 1]]  # 22: 检测头
```

### 2.2 三大组成部分

| 部分 | 层索引 | 主要模块 | 参数量 | 占比 | 作用 |
|------|--------|----------|--------|------|------|
| **Backbone** | model.0-9 | Conv, C2f, SPPF | 5,079,712 | 44.5% | 特征提取 (从输入图像提取多尺度特征) |
| **Neck** | model.10-21 | Upsample, Concat, C2f, Conv | 3,939,840 | 34.5% | 特征融合 (FPN+PAN 结构) |
| **Head** | model.22 | OBB | 2,402,614 | 21.0% | 检测输出 (分类 + 框回归 + 角度) |

---

## 3. 各层详细信息

```
层索引    类型          部分        参数量          作用
─────────────────────────────────────────────────────────────
model.0   Conv        Backbone      928         下采样/特征提取
model.1   Conv        Backbone     18,560       下采样/特征提取
model.2   C2f         Backbone     29,056       特征学习模块
model.3   Conv        Backbone     73,984       下采样/特征提取
model.4   C2f         Backbone    197,632       特征学习模块
model.5   Conv        Backbone    295,424       下采样/特征提取
model.6   C2f         Backbone    788,480       特征学习模块
model.7   Conv        Backbone  1,180,672       下采样/特征提取
model.8   C2f         Backbone  1,838,080       特征学习模块
model.9   SPPF        Backbone    656,896       空间金字塔池化 (感受野增强)
─────────────────────────────────────────────────────────────
model.10  Upsample    Neck             0        上采样 (特征图放大)
model.11  Concat      Neck             0        特征拼接融合
model.12  C2f         Neck       591,360       特征学习模块
model.13  Upsample    Neck             0        上采样
model.14  Concat      Neck             0        特征拼接融合
model.15  C2f         Neck       148,224       特征学习模块 (P3/8-small)
model.16  Conv        Neck       147,712       下采样
model.17  Concat      Neck             0        特征拼接融合
model.18  C2f         Neck       493,056       特征学习模块 (P4/16-medium)
model.19  Conv        Neck       590,336       下采样
model.20  Concat      Neck             0        特征拼接融合
model.21  C2f         Neck     1,969,152       特征学习模块 (P5/32-large)
─────────────────────────────────────────────────────────────
model.22  OBB         Head     2,402,614       OBB检测头 (分类+框+角度)
```

---

## 4. freeze 参数详解

### 4.1 freeze 参数含义

`freeze=N` 表示冻结前 N 个顶层模块 (model.0 到 model.N-1)，被冻结的层参数不更新。

### 4.2 freeze 对照表

| freeze值 | 冻结范围 | 冻结参数占比 | 效果 |
|----------|----------|--------------|------|
| `freeze=0` | 不冻结 | 0% | 全模型训练 |
| `freeze=5` | model.0-4 | 2.8% | 冻结浅层特征 (部分 Backbone) |
| `freeze=10` | model.0-9 | **44.5%** | 冻结完整 Backbone |
| `freeze=15` | model.0-14 | 49.6% | Backbone + 部分 Neck |
| `freeze=22` | model.0-21 | 79.0% | 只训练检测头 (Head) |

### 4.3 微调场景建议

| 场景 | 推荐 freeze 值 | 学习率建议 |
|------|----------------|------------|
| 新数据量少，数据分布相似 | `freeze=10` 或 `freeze=15` | `lr0=0.001` 或更低 |
| 新数据量中等 | `freeze=5` 或 `freeze=10` | `lr0=0.001` |
| 新数据量大，或数据分布差异大 | `freeze=0` 或 `freeze=5` | `lr0=0.01` |

---

## 5. 常用查询代码

### 5.1 查看模型摘要

```python
from ultralytics import YOLO

model = YOLO('your_model.pt')
model.info()
```

### 5.2 遍历各层

```python
from ultralytics import YOLO

model = YOLO('your_model.pt')

for i, module in enumerate(model.model.model):
    params = sum(p.numel() for p in module.parameters())
    print(f"model.{i}: {module.__class__.__name__}, {params:,} params")
```

### 5.3 判断层所属部分

```python
for i, module in enumerate(model.model.model):
    if i <= 9:
        part = 'Backbone'
    elif i <= 21:
        part = 'Neck'
    else:
        part = 'Head'
    print(f"model.{i}: {part}")
```

### 5.4 微调训练示例

```python
from ultralytics import YOLO

# 加载已训练模型
model = YOLO('output/models/yolov8s-obb/exp_v17/weights/best.pt')

# 微调训练
model.train(
    data='datasets/merged_dataset/dataset.yaml',
    epochs=100,
    lr0=0.001,        # 降低学习率
    freeze=10,        # 冻结 Backbone
    batch=8,
    imgsz=640,
)
```

---

## 6. 关键概念

### 6.1 层数统计差异

- **官方层数 (145)**: 只计算有参数的主要层 (Conv, C2f, SPPF, OBB 等)
- **实际子层数 (265)**: 包含所有子模块 (Conv 内部的 conv, bn, act 等都单独计算)

### 6.2 Backbone / Neck / Head 作用

```
输入图像
    │
    ▼
┌─────────────────────────────────────┐
│  Backbone (特征提取)                 │
│  - 逐层下采样，提取多尺度特征         │
│  - P1→P2→P3→P4→P5 (尺寸递减)         │
│  - 通用特征，适合冻结                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Neck (特征融合)                     │
│  - FPN: 自顶向下传递语义信息          │
│  - PAN: 自底向上传递定位信息          │
│  - 融合多尺度特征                    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Head (检测输出)                     │
│  - 分类分支: 预测类别                │
│  - 框回归分支: 预测边界框            │
│  - 角度分支 (OBB): 预测旋转角度       │
└─────────────────────────────────────┘
    │
    ▼
检测结果
```

---

## 7. 相关文件路径

| 文件 | 路径 |
|------|------|
| YOLOv8-OBB 模型配置 | `ultralytics/cfg/models/v8/yolov8-obb.yaml` |
| YOLOv11-OBB 模型配置 | `ultralytics/cfg/models/11/yolo11-obb.yaml` |
| 训练默认参数 | `ultralytics/cfg/default.yaml` |
| 项目配置 | `YOLO/configs/default.yaml` |

---

*笔记创建时间: 2026-02-25*
