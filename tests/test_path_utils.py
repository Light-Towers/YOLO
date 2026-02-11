"""
测试路径工具模块
"""
import sys
from pathlib import Path

import pytest

from src.utils.path_utils import (
    ensure_absolute,
    ensure_path,
    get_image_files,
    get_project_root,
    safe_mkdir,
)


class TestPathUtils:
    """路径工具测试"""

    def test_get_project_root(self):
        """测试获取项目根目录"""
        root = get_project_root()
        assert root.exists()
        assert (root / "README.md").exists() or (root / ".git").exists()

    def test_ensure_path_with_string(self, tmp_path):
        """测试确保路径（字符串输入）"""
        result = ensure_path(str(tmp_path))
        assert isinstance(result, Path)
        assert result == tmp_path

    def test_ensure_path_with_path(self, tmp_path):
        """测试确保路径（Path 对象输入）"""
        result = ensure_path(tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path

    def test_ensure_absolute_with_absolute_path(self):
        """测试确保绝对路径（已经是绝对路径）"""
        absolute = Path("/tmp/test")
        result = ensure_absolute(absolute)
        assert result == absolute

    def test_ensure_absolute_with_relative_path(self, tmp_path):
        """测试确保绝对路径（相对路径）"""
        relative = Path("subdir/file.txt")
        result = ensure_absolute(relative, tmp_path)
        assert result.is_absolute()
        assert str(tmp_path) in str(result)

    def test_safe_mkdir(self, tmp_path):
        """测试安全创建目录"""
        new_dir = tmp_path / "new_dir" / "sub_dir"
        result = safe_mkdir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_safe_mkdir_existing_dir(self, tmp_path):
        """测试安全创建已存在的目录"""
        result = safe_mkdir(tmp_path)
        assert result == tmp_path

    def test_get_image_files(self, tmp_path, sample_image):
        """测试获取图片文件"""
        images = get_image_files(tmp_path, recursive=False)
        assert len(images) > 0
        assert sample_image in images

    def test_get_image_files_recursive(self, tmp_path):
        """测试递归获取图片文件"""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "test.jpg").touch()
        (tmp_path / "test.png").touch()

        images = get_image_files(tmp_path, recursive=True)
        assert len(images) == 2

    def test_get_image_files_filter_extensions(self, tmp_path):
        """测试获取图片文件（过滤扩展名）"""
        (tmp_path / "test.jpg").touch()
        (tmp_path / "test.png").touch()
        (tmp_path / "test.txt").touch()
        (tmp_path / "test.doc").touch()

        images = get_image_files(tmp_path, recursive=False)
        assert len(images) == 2
        assert all(img.suffix in ['.jpg', '.png'] for img in images)
