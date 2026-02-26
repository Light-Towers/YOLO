# 训练日志分析

## 训练配置
- **模型**: yolo11m-obb-20260211.pt
- **数据集**: booth_final_merged_20260226_1530
- **训练轮数**: 100 epochs
- **图像尺寸**: 1024x1024
- **冻结层数**: 10 (model.0-9)
- **设备**: Tesla V100-SXM2-32GB (CUDA:0)

---

## 问题分析

### 1. ⚠️ 学习率设置问题

**日志输出**:
```
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.002, momentum=0.9)
```

**问题**:
- 使用 `optimizer='auto'` 时，系统会忽略手动设置的 `lr0` 和 `momentum`
- 系统自动选择了 `lr=0.002`，比微调推荐的 `0.001` 略高

**解决方案**:
```python
# 修改前
optimizer='auto',
lr0=0.01,

# 修改后
optimizer='AdamW',  # 明确指定优化器
lr0=0.01,  # 微调时改为 0.001 或更低
```

---

### 2. ✅ 冻结层情况

**冻结的层**:
- model.0-9: Backbone 层
- model.23: DFL (Distribution Focal Loss) 层

**冻结参数**: 691 个参数项

**各部分参数分布**:
| 部分 | 层索引 | 状态 |
|------|--------|------|
| Backbone | model.0-9 | 冻结 |
| Neck | model.10-22 | 训练 |
| Head (DFL) | model.23 | 冻结 |

---

### 3. ✅ 数据集统计

| 分割 | 图片数 | 标注数 | 背景 |
|------|--------|--------|------|
| Train | 377 | ~? | 114 |
| Val | 84 | 1825 | 0 |

**备注**:
- 训练集有 114 个无标注的背景图片
- 验证集 mAP 接近 1.0，说明数据质量很好

---

### 4. ✅ 初始性能

**Epoch 1/100**:
```
Box(P) = 0.979
R      = 0.972
mAP50  = 0.99
mAP50-95 = 0.981
```

**Epoch 2/100**:
```
Box(P) = 0.982
R      = 0.958
mAP50  = 0.99
mAP50-95 = 0.983
```

**分析**:
- 预训练模型在该数据集上表现已经非常好
- mAP 接近 1.0，提升空间有限
- 可能数据分布与预训练时使用的训练数据高度相似

---

### 5. ⚠️ GPU 内存使用

```
AutoBatch: Using batch-size 8 for CUDA:0 13.13G/16.00G (82% ✅
```

**配置**: `batch=0.9` (90% GPU 内存)

**实际**: 自动选择 batch_size=8

**内存占用**: 13.13GB / 16GB = 82%

---

## 模型架构信息

```
YOLO11m-obb summary:
- Layers: 247
- Parameters: 20,902,614 (~20.9M)
- GFLOPs: 71.9 @ 1024x1024
```

---

## 优化建议

### 1. 修改学习率设置

**train.py 第 76-80 行**:
```python
# ========== 优化器与学习率 ==========
optimizer='AdamW',  # 不使用 'auto'
lr0=TRAINING_CONSTANTS.DEFAULT_LR,  # 微调时改为 0.001
lrf=0.01,
momentum=TRAINING_CONSTANTS.DEFAULT_MOMENTUM,
```

### 2. 微调时的推荐参数

| 参数 | 原始训练 | 微调（数据多） | 微调（数据少） |
|------|----------|----------------|----------------|
| lr0 | 0.01 | 0.001 | 0.0001 |
| lrf | 0.01 | 0.1 | 0.1 |
| epochs | 300 | 100 | 50-100 |
| freeze | 0 | 10 | 10-15 |

### 3. 鉴于当前性能已接近完美

**建议**:
- 如果当前性能已经满足需求，无需继续微调
- 如需进一步提升，考虑：
  1. 增加数据多样性
  2. 尝试更大模型（yolo11l-obb, yolo11x-obb）
  3. 测试数据增强策略

---

## 学习率对照表

| freeze值 | 冻结范围 | 冻结参数占比 | 推荐学习率 |
|----------|----------|--------------|------------|
| freeze=0 | 不冻结 | 0% | 0.01 |
| freeze=5 | model.0-4 | 2.8% | 0.005 |
| freeze=10 | model.0-9 | 44.5% | 0.001 |
| freeze=15 | model.0-14 | 49.6% | 0.001 |

---

*日志分析时间: 2026-02-26*
