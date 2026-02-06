# YOLO 展位检测项目

基于 YOLO 的展位检测系统，支持目标检测（OBB）、实例分割和切片推理。

## 功能特性

- 🔍 **多任务支持**：OBB 检测、实例分割
- 📊 **SAHI 切片推理**：支持大图检测
- 🏷️ **标注转换**：LabelMe ↔ YOLO 格式
- 🔪 **图像切分**：智能数据集切分工具
- 📈 **训练流水线**：完整的训练-验证-预测流程
- ⚙️ **配置管理**：YAML 配置文件支持

## 快速开始

### 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 或使用 pyproject.toml
pip install -e .
```

### 项目结构

```
YOLO/
├── src/                      # 源代码
│   ├── core/                # 核心模块（配置、异常、常量）
│   ├── training/            # 训练相关
│   ├── inference/           # 推理相关
│   ├── data/                # 数据处理
│   └── utils/               # 工具函数
├── configs/                  # 配置文件
├── script/                   # 脚本文件（兼容旧版）
├── datasets/                 # 数据集
├── models/                   # 预训练模型
├── output/                   # 输出结果
├── tests/                    # 测试代码
└── logs/                     # 日志文件
```

### 训练模型

```bash
# 使用训练脚本
python script/train.py \
    --model yolov8s-obb.pt \
    --dataset booth_seg \
    --epochs 300

# 或使用配置文件
python script/train.py --config configs/default.yaml
```

### 推理预测

```bash
# SAHI 切片推理
python script/predict_sahi.py \
    --model output/models/yolov8s-obb/best.pt \
    --image images/2024年展位图.jpg

# OBB 推理
python script/predict_obb.py \
    --model output/models/yolov8s-obb/best.pt \
    --source images/
```

### 训练预测流水线

```bash
# 运行完整的训练-预测流水线
python script/train_predict_pipeline.py
```

## 配置说明

主配置文件位于 `configs/default.yaml`，包含：

- **dataset**: 数据集配置
- **training**: 训练参数
- **inference**: 推理参数
- **paths**: 路径配置
- **logging**: 日志配置

示例：

```yaml
training:
  models:
    - "yolov8s-obb.pt"
    - "yolo11s-obb.pt"
  epochs: 300
  patience: 50
  batch: 0.9
```

## 开发

### 代码格式化

```bash
# 格式化代码
black src/ script/

# 排序导入
isort src/ script/
```

### 类型检查

```bash
mypy src/
```

### 运行测试

```bash
pytest tests/ -v --cov=src
```

## 常见问题

### 1. 模型路径问题

使用 `get_model_path()` 函数自动查找模型路径：

```python
from src.utils.path_utils import get_project_root

project_root = get_project_root()
model_path = project_root / "models" / "yolov8" / "yolov8s-obb.pt"
```

### 2. 数据集路径问题

使用 `update_dataset_path()` 更新 dataset.yaml：

```python
from script.train import update_dataset_path

update_dataset_path(dataset_yaml_path, dataset_root)
```

### 3. GPU 内存不足

调整 batch size 或使用自动分配：

```yaml
training:
  batch: 0.9  # 自动分配 90% 显存
  # 或
  batch: -1   # 自动计算最大可用 batch
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
