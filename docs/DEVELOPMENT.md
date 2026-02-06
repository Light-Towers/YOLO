# 开发指南

## 环境设置

### 快速设置

```bash
# 使用设置脚本
bash setup_dev.sh

# 或手动设置
pip install -r requirements.txt
pip install -e ".[dev]"
pre-commit install
```

## 代码风格

### 格式化代码

```bash
# 使用 Black 格式化代码
black src/ script/

# 使用 isort 排序导入
isort src/ script/
```

### 代码检查

```bash
# 使用 flake8 检查代码
flake8 src/ script/

# 使用 mypy 进行类型检查
mypy src/
```

## 测试

### 运行所有测试

```bash
pytest tests/ -v
```

### 运行特定测试

```bash
# 运行特定测试文件
pytest tests/test_config.py -v

# 运行特定测试类
pytest tests/test_config.py::TestConfig -v

# 运行特定测试方法
pytest tests/test_config.py::TestConfig::test_default_config -v
```

### 生成测试覆盖率报告

```bash
# 生成 HTML 报告
pytest tests/ --cov=src --cov-report=html

# 生成终端报告
pytest tests/ --cov=src --cov-report=term
```

## 配置管理

### 加载配置

```python
from src.core import load_config

# 加载默认配置
config = load_config()

# 加载自定义配置
config = load_config("configs/custom.yaml")

# 访问配置
print(config.training.epochs)
print(config.dataset.name)
```

### 使用配置

```python
from src.core import load_config

config = load_config()

# 训练
model.train(
    epochs=config.training.epochs,
    batch=config.training.batch,
    imgsz=config.training.imgsz,
)

# 推理
detection_model = AutoDetectionModel.from_pretrained(
    confidence_threshold=config.inference.confidence_threshold,
    device=get_device(config.training.device),
)
```

## 日志使用

### 获取 Logger

```python
from src.utils import get_logger

logger = get_logger("my_module")
logger.info("这是一条信息日志")
logger.warning("这是一条警告日志")
logger.error("这是一条错误日志")
```

### 配置日志级别

```python
from src.utils import get_logger

# 设置日志级别为 DEBUG
logger = get_logger("my_module", level="DEBUG")
```

### 保存日志到文件

```python
from pathlib import Path
from src.utils import get_logger

log_file = Path("logs/my_module.log")
logger = get_logger("my_module", log_file=log_file)
```

## 路径工具

### 获取项目根目录

```python
from src.utils import get_project_root

root = get_project_root()
```

### 获取图片文件

```python
from src.utils import get_image_files

# 获取目录中所有图片（不递归）
images = get_image_files(Path("images"), recursive=False)

# 递归获取所有图片
images = get_image_files(Path("images"), recursive=True)
```

### 安全创建目录

```python
from src.utils import safe_mkdir

safe_mkdir(Path("output/new_dir"))
```

## 异常处理

### 自定义异常

```python
from src.core.exceptions import YOLOProjectError, ModelNotFoundError

# 抛出异常
if not model_path.exists():
    raise ModelNotFoundError(
        f"模型文件不存在: {model_path}\n"
        f"请检查路径或下载模型"
    )

# 捕获异常
try:
    result = some_function()
except YOLOProjectError as e:
    logger.error(f"项目错误: {e}")
except Exception as e:
    logger.error(f"未知错误: {e}")
```

## 添加新功能

### 1. 创建新模块

```bash
# 在 src/ 下创建新模块
mkdir -p src/new_module
touch src/new_module/__init__.py
```

### 2. 编写代码

```python
# src/new_module/__init__.py
from .my_class import MyClass

__all__ = ['MyClass']
```

### 3. 编写测试

```python
# tests/test_new_module.py
import pytest
from src.new_module import MyClass

def test_my_class():
    obj = MyClass()
    assert obj is not None
```

### 4. 运行测试

```bash
pytest tests/test_new_module.py -v
```

## Git 工作流

### Pre-commit 检查

```bash
# 手动运行所有检查
pre-commit run --all-files

# 只检查已修改的文件
pre-commit run
```

### 提交代码

```bash
# 添加文件
git add .

# 运行 pre-commit
pre-commit run

# 提交
git commit -m "feat: 添加新功能"
```

### 分支命名

- `feature/xxx` - 新功能
- `fix/xxx` - 错误修复
- `docs/xxx` - 文档更新
- `refactor/xxx` - 代码重构
- `test/xxx` - 测试相关

## 常见问题

### 1. 导入错误

确保项目根目录在 Python 路径中：

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. 类型检查错误

对于第三方库，可以忽略类型检查：

```python
# 使用 type: ignore
from some_lib import some_func  # type: ignore
```

### 3. 测试失败

查看详细错误信息：

```bash
pytest tests/ -v -s
```
