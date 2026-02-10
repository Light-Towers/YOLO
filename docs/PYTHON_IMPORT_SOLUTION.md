# Python 模块导入问题解决笔记

## 问题现象

运行脚本时出现 `ModuleNotFoundError: No module named 'src'` 错误：

```bash
$ python script/predict.py
ModuleNotFoundError: No module named 'src'
```

## 问题原因

**项目结构：**
```
YOLO/                      ← 项目根目录
├── src/                   ← Python包（需要__init__.py）
│   ├── core/
│   ├── utils/
│   └── __init__.py        ← 最初缺少此文件
├── script/
│   └── predict.py         ← 从此处运行脚本
└── pyproject.toml
```

**根本原因：**
1. 在 `YOLO/script/` 子目录运行脚本，Python 的模块搜索路径不包含 `YOLO/` 根目录
2. `src` 目录缺少 `__init__.py` 文件，Python 无法识别其为包
3. `pyproject.toml` 缺少构建配置，`pip install -e .` 无法正确安装

---

## 解决方案

### 方案对比

| 方案 | 方法 | 优缺点 |
|------|------|--------|
| A | 修改脚本添加 `sys.path` | ❌ 每个脚本都要改，冗余 |
| B | **Editable 安装（推荐）** | ✅ 一劳永逸，最优雅 |
| C | 设置 `PYTHONPATH` 环境变量 | ⚠️ 每次运行前都要设置 |
| D | 从根目录用模块方式运行 | ⚠️ 运行命令较繁琐 |

### 最终方案：Editable 安装

#### 步骤1：修改 `pyproject.toml` 添加构建配置

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[project]
name = "yolo-booth-detection"
version = "0.1.0"
# ... 其他配置
```

#### 步骤2：创建 `src/__init__.py`

```python
# YOLO Booth Detection - Source Package

try:
    from importlib.metadata import version
    __version__ = version("yolo-booth-detection")
except Exception:
    __version__ = "0.1.0"  # 备用默认值
```

#### 步骤3：执行 Editable 安装

```bash
cd /home/aistudio/YOLO
pip install -e .
```

#### 步骤4：验证安装

```bash
python -c "import src; print(src.__file__)"
# 输出: /home/aistudio/YOLO/src/__init__.py
```

---

## 关键知识点

### 1. `__init__.py` 的作用

**功能：**
- 标识目录为 Python 包，使其可 `import`
- 包导入时自动执行初始化代码
- 定义 `__all__` 控制 `from package import *` 的导出内容

**层级关系：**
```
src/                       ← 根包
├── __init__.py            ← 标识 src 是包
├── core/                  ← 子包
│   ├── __init__.py        ← 标识 core 是子包
│   └── config.py
└── utils/                 ← 子包
    ├── __init__.py        ← 标识 utils 是子包
    └── file_utils.py
```

**常见用途：**
```python
# __init__.py 中可以写：

# 1. 版本号（动态读取）
from importlib.metadata import version
__version__ = version("package-name")

# 2. 简化导入（让用户更方便）
from src.utils import get_device, get_logger
from src.core import load_config

__all__ = ['get_device', 'get_logger', 'load_config']

# 3. 初始化代码
import logging
logging.basicConfig(level=logging.INFO)
```

### 2. Editable 安装详解

**是什么？**
- 只在 Python 环境创建一个**指向源代码的链接**
- 修改源码**立即生效**，无需重新安装

**vs 普通安装：**

| 普通安装 | Editable 安装 |
|---------|--------------|
| `pip install .` | `pip install -e .` |
| 代码复制到 Python 环境 | 创建符号链接指向源码 |
| 修改后需重新安装 | 修改立即生效 |
| 适合最终用户 | 适合开发时使用 |

### 3. `egg-info` 目录

**位置：** `src/yolo_booth_detection.egg-info/`

**作用：** 记录包的元数据信息

**包含文件：**
| 文件 | 内容 |
|------|------|
| `PKG-INFO` | 包名称、版本、依赖、README |
| `top_level.txt` | 顶层模块列表（core, utils） |
| `requires.txt` | 依赖列表 |
| `SOURCES.txt` | 源代码文件列表 |

**注意：** 可以删除，不影响功能，下次 pip 命令会重新生成。

### 4. 版本号管理最佳实践

**推荐方式：** 单点维护，多处使用

```toml
# pyproject.toml（唯一真实来源）
[project]
version = "0.1.0"
```

```python
# src/__init__.py（动态读取）
from importlib.metadata import version
__version__ = version("yolo-booth-detection")
```

```python
# 脚本中使用
from src import __version__
print(f"当前版本: {__version__}")
```

---

## 运行脚本方式

安装成功后，**从任何目录**都可以运行：

```bash
# 方式1：在项目根目录（推荐）
cd /home/aistudio/YOLO
python script/predict.py

# 方式2：在 script 目录
cd /home/aistudio/YOLO/script
python predict.py

# 方式3：从任意目录
python /home/aistudio/YOLO/script/predict.py
```

---

## 总结

1. **Editable 安装**是解决 Python 项目模块导入问题的最佳方案
2. `pyproject.toml` 需要完整的 `[build-system]` 配置
3. 每个包目录都需要 `__init__.py` 文件
4. 版本号应在 `pyproject.toml` 中单点维护，通过 `importlib.metadata` 动态读取
5. 安装成功后，脚本可以从任意目录运行
