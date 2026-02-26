# YOLO 训练优化器详解

## 支持的优化器类型

| 优化器 | 全称 | 说明 | 适用场景 |
|--------|------|------|----------|
| **SGD** | Stochastic Gradient Descent | 随机梯度下降 | 大规模数据，需要泛化能力 |
| **Adam** | Adaptive Moment Estimation | 自适应矩估计 | 大多数任务，平衡性好 |
| **AdamW** | Adam with decoupled weight decay | 带权重衰减的 Adam | 推荐默认选择 |
| **Adamax** | Adam with infinity norm | 基于无穷范数的 Adam | 稀疏数据，大模型 |
| **NAdam** | Nesterov-accelerated Adam | Nesterov 加速 Adam | 需要更平滑收敛 |
| **RAdam** | Rectified Adam | 整流 Adam | 训练初期不稳定时 |
| **RMSProp** | Root Mean Square Propagation | 均方根传播 | RNN，非平稳目标 |
| **auto** | 自动选择 | 根据任务自动选择 | 不确定时使用 |

---

## 各优化器详解

### 1. SGD (Stochastic Gradient Descent)

**特点**：
- 最基础的优化算法
- 简单、高效、泛化能力强
- 需要手动调整学习率

**公式**：
```
v_t = momentum * v_{t-1} + lr * ∇L(w_t)
w_{t+1} = w_t - v_t
```

**优点**：
- ✅ 泛化能力强
- ✅ 适合大规模数据集
- ✅ 内存占用小

**缺点**：
- ❌ 收敛速度慢
- ❌ 需要精细调参

**推荐场景**：
- 大规模数据集 (100K+ 图片)
- 追求数据泛化能力
- 从零训练

**参数设置**：
```python
optimizer='SGD',
lr0=0.01,
momentum=0.937,
weight_decay=0.0005,
```

---

### 2. Adam (Adaptive Moment Estimation)

**特点**：
- 自适应学习率
- 结合了一阶矩（均值）和二阶矩（方差）
- 收敛快，泛化略弱于 SGD

**公式**：
```
m_t = β1 * m_{t-1} + (1-β1) * ∇L
v_t = β2 * v_{t-1} + (1-β2) * ∇L²
w_{t+1} = w_t - lr * m_t / (√v_t + ε)
```

**优点**：
- ✅ 收敛速度快
- ✅ 对初始学习率不敏感
- ✅ 适合大多数任务

**缺点**：
- ❌ 泛化能力略弱于 SGD
- ❌ 可能陷入局部最优

**推荐场景**：
- 中等规模数据集
- 快速迭代
- 微调训练

**参数设置**：
```python
optimizer='Adam',
lr0=0.001,  # Adam 通常用更小的学习率
momentum=0.937,  # 不常用，但兼容
weight_decay=0.0005,
```

---

### 3. AdamW (Adam with decoupled weight decay) ⭐ 推荐

**特点**：
- Adam 的改进版本
- 将权重衰减从梯度更新中解耦
- 更好的正则化效果
- **Ultralytics 自动选择时的默认值**

**与 Adam 的区别**：
```python
# Adam (权重衰减直接叠加到梯度)
w_{t+1} = w_t - lr * (m_t/√v_t + λ * w_t)

# AdamW (权重衰减独立应用)
w'_{t+1} = w_t - lr * (m_t/√v_t)
w_{t+1} = w'_{t+1} - lr * λ * w'_{t+1}
```

**优点**：
- ✅ 继承 Adam 的快速收敛
- ✅ 更好的正则化效果
- ✅ 减少过拟合
- ✅ YOLO 推荐使用

**缺点**：
- ❌ 相对较新，某些旧框架不支持

**推荐场景**：
- **几乎所有 YOLO 训练任务**
- 微调训练
- 中小规模数据集

**参数设置**：
```python
optimizer='AdamW',
lr0=0.001,  # 微调用 0.001，从零训练用 0.01
momentum=0.9,
weight_decay=0.0005,
```

---

### 4. Adamax

**特点**：
- Adam 的变体，使用无穷范数
- 对梯度异常值更鲁棒
- 适合稀疏数据

**优点**：
- ✅ 数值稳定性好
- ✅ 适合非平稳目标

**缺点**：
- ❌ 收敛可能较慢
- ❌ 应用场景较少

**推荐场景**：
- 稀疏数据
- 大模型训练

---

### 5. NAdam (Nesterov-accelerated Adam)

**特点**：
- Adam + Nesterov 动量
- 更激进的梯度预判
- 收敛更平滑

**优点**：
- ✅ 收敛速度更快
- ✅ 更稳定的收敛过程

**缺点**：
- ❌ 可能训练初期不稳定

**推荐场景**：
- 需要快速收敛
- 数据质量高

---

### 6. RAdam (Rectified Adam)

**特点**：
- 整流 Adam，解决训练初期不稳定问题
- 自动调整学习率

**优点**：
- ✅ 训练初期更稳定
- ✅ 自适应学习率调整

**缺点**：
- ❌ 计算略复杂

**推荐场景**：
- 训练初期不稳定
- 小数据集

---

### 7. RMSProp

**特点**：
- 基于梯度的移动平均
- 适合 RNN
- 对非平稳目标鲁棒

**优点**：
- ✅ 适合序列数据
- ✅ 对稀疏梯度鲁棒

**缺点**：
- ❌ 通常不如 Adam
- ❌ 超参数敏感

**推荐场景**：
- RNN/LSTM
- 时序数据

---

### 8. auto

**特点**：
- Ultralytics 自动选择优化器
- 根据任务类型和数据集大小决策

**自动选择规则**：
```python
if 从零训练:
    return 'SGD' if 大规模数据集 else 'AdamW'
else:  # 微调
    return 'AdamW'
```

**注意**：
- ⚠️ 使用 `auto` 时，会**忽略手动设置的 lr0**，自动计算学习率

---

## 优化器对比

| 优化器 | 收敛速度 | 泛化能力 | 内存占用 | 调参难度 | 推荐度 |
|--------|----------|----------|----------|----------|--------|
| SGD | 慢 | ⭐⭐⭐⭐⭐ | 低 | 困难 | ⭐⭐⭐ |
| Adam | 快 | ⭐⭐⭐ | 中 | 简单 | ⭐⭐⭐⭐ |
| **AdamW** | 快 | ⭐⭐⭐⭐ | 中 | 简单 | ⭐⭐⭐⭐⭐ |
| Adamax | 中 | ⭐⭐⭐ | 中 | 简单 | ⭐⭐⭐ |
| NAdam | 很快 | ⭐⭐⭐ | 中 | 简单 | ⭐⭐⭐⭐ |
| RAdam | 快 | ⭐⭐⭐ | 中 | 简单 | ⭐⭐⭐ |
| RMSProp | 中 | ⭐⭐ | 中 | 中 | ⭐⭐ |

---

## 参数说明

### lr0 (初始学习率)

| 优化器 | 从零训练 | 微调 | 仅训练 Head |
|--------|----------|------|------------|
| SGD | 0.01 | 0.001-0.005 | 0.0001 |
| Adam | 0.001 | 0.0001-0.0005 | 0.00001 |
| AdamW | 0.01 | 0.001 | 0.0001 |

### momentum (动量)

- 仅用于 SGD、RMSProp
- 典型值：0.9 - 0.95
- YOLO 默认：0.937

### weight_decay (权重衰减)

- L2 正则化强度
- 典型值：0.0001 - 0.001
- YOLO 默认：0.0005

---

## 推荐配置

### 场景 1: 从零训练 (大规模数据集)

```python
model.train(
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    epochs=300,
)
```

### 场景 2: 从零训练 (中小规模数据集)

```python
model.train(
    optimizer='AdamW',
    lr0=0.01,
    weight_decay=0.0005,
    epochs=300,
)
```

### 场景 3: 微调训练 (数据多)

```python
model.train(
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    freeze=10,
    epochs=100,
)
```

### 场景 4: 微调训练 (数据少)

```python
model.train(
    optimizer='AdamW',
    lr0=0.0001,
    weight_decay=0.0001,
    freeze=10,
    epochs=50,
)
```

---

## Optimizer 的核心作用

**Optimizer（优化器）** 是训练过程中的"导航员"或计算引擎。

它的核心作用包括：

- **更新模型权重**：优化器通过 [反向传播](https://www.ultralytics.com/glossary/backpropagation) 计算出的梯度，自动调整模型的权重和偏置，以减少预测值与真实值之间的误差。
- **最小化损失函数**：它的目标是寻找 [损失函数](https://www.ultralytics.com/glossary/loss-function) 的最小值。你可以将其想象成一个向导，决定了模型在复杂的"误差地形"中下降的方向和步长（[学习率](https://www.ultralytics.com/glossary/learning-rate)）。
- **平衡速度与精度**：不同的 [优化算法](https://www.ultralytics.com/glossary/optimization-algorithm)（如 **SGD** 或 **AdamW**）通过不同的策略（如动量或自适应学习率）来加快收敛速度，并帮助模型跳出局部最优解，从而获得更好的泛化性能。

---

## 主要区别

| 特性 | **Adam/AdamW** | **SGD** | **RMSProp** |
|------|----------------|----------|-------------|
| **收敛速度** | ⭐⭐⭐⭐ 最快 | ⭐⭐ 慢 | ⭐⭐⭐ 中等 |
| **泛化能力** | ⭐⭐⭐ 良好 | ⭐⭐⭐⭐⭐ 优秀 | ⭐⭐⭐ 良好 |
| **调参难度** | 简单 | 困难 | 中等 |
| **适用任务** | 大多数任务 | 大规模数据集 | RNN/时序数据 |

- **Adam/AdamW**：通常收敛最快，对初始学习率不敏感，非常适合快速实验。
- **SGD**：虽然收敛较慢且需要更多调参，但在训练后期通常能找到更优的局部最小值，提升模型在测试集上的表现。
- **RMSProp**：常用于处理循环神经网络或极其不稳定的梯度。

---

## 注意事项

1. **避免使用 `auto`**：会覆盖手动设置的学习率
2. **Adam vs AdamW**：优先使用 AdamW
3. **微调用小学习率**：比从零训练小 10-100 倍
4. **大冻结学习率更小**：freeze 越大，lr0 越小
5. **AdamW 是标准选择**：目前训练 Ultralytics YOLO26 等先进视觉模型的标准选择

---

*优化器选择指南 - 2026-02-26*
