"""
路径工具模块
统一项目中的路径处理逻辑
"""
import shutil
from pathlib import Path
from typing import Optional, Union

# 类型别名
PathLike = Union[str, Path]


class ProjectPaths:
    """项目路径配置类（单例模式）"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return
        self.root = self._find_root()
        self.data = self.root / "datasets"
        self.models = self.root / "models"
        self.outputs = self.root / "output"
        self.logs = self.root / "logs"
        self.images = self.root / "images"
        self.configs = self.root / "configs"
        self.initialized = True

    def _find_root(self) -> Path:
        """自动查找项目根目录"""
        current = Path(__file__).resolve()
        for _ in range(5):
            if (current / ".git").exists() or (current / "README.md").exists():
                return current
            current = current.parent
        return Path(__file__).resolve().parent.parent.parent

    @property
    def labelme_annotations(self) -> Path:
        """LabelMe标注文件目录"""
        return self.root / "annotations"

    @property
    def training_output(self) -> Path:
        """训练输出目录"""
        return self.outputs / "models"

    @property
    def inference_output(self) -> Path:
        """推理输出目录"""
        return self.outputs / "results"


def ensure_path(path: PathLike) -> Path:
    """确保返回Path对象"""
    return Path(path) if not isinstance(path, Path) else path


def get_relative_path(path: PathLike, base: PathLike) -> Path:
    """获取相对路径"""
    path = ensure_path(path)
    base = ensure_path(base)
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def ensure_absolute(path: PathLike, base: Optional[PathLike] = None) -> Path:
    """确保返回绝对路径"""
    path = ensure_path(path)
    if path.is_absolute():
        return path

    if base is not None:
        return ensure_path(base) / path

    return path.resolve()


def safe_mkdir(path: PathLike, parents: bool = True) -> Path:
    """安全创建目录"""
    path = ensure_path(path)
    path.mkdir(parents=parents, exist_ok=True)
    return path


def safe_remove(path: PathLike) -> bool:
    """安全删除文件或目录"""
    path = ensure_path(path)
    if not path.exists():
        return False

    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)

    return True


def get_image_files(directory: PathLike, recursive: bool = False) -> list[Path]:
    """获取目录中的所有图片文件"""
    from src.core.constants import IMAGE_CONSTANTS

    directory = ensure_path(directory)
    extensions = IMAGE_CONSTANTS.SUPPORTED_FORMATS
    files = []

    if recursive:
        for ext in extensions:
            files.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))

    return sorted(set(files))


def get_file_size(path: PathLike) -> int:
    """获取文件大小（字节）"""
    path = ensure_path(path)
    if not path.is_file():
        raise ValueError(f"路径不是文件: {path}")
    return path.stat().st_size


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小为可读字符串"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_project_root() -> Path:
    """获取项目根目录"""
    return ProjectPaths().root
