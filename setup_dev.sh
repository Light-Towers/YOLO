#!/bin/bash
# 开发环境设置脚本

set -e

echo "====================================="
echo "YOLO 项目开发环境设置"
echo "====================================="

# 1. 创建虚拟环境（可选）
if [ "$1" == "--venv" ]; then
    echo "创建虚拟环境..."
    python -m venv .venv
    source .venv/bin/activate
fi

# 2. 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt

# 3. 安装开发依赖
echo "安装开发依赖..."
pip install -e ".[dev]"

# 4. 安装 pre-commit hooks
echo "安装 pre-commit hooks..."
pre-commit install

# 5. 创建必要的目录
echo "创建项目目录..."
mkdir -p logs
mkdir -p output
mkdir -p output/results
mkdir -p output/models
mkdir -p datasets
mkdir -p tests/fixtures

echo ""
echo "====================================="
echo "开发环境设置完成！"
echo "====================================="
echo ""
echo "可用命令："
echo "  运行训练: python script/train.py"
echo "  运行预测: python script/predict_sahi.py"
echo "  运行测试: pytest tests/ -v"
echo "  代码格式化: black src/ script/"
echo "  类型检查: mypy src/"
echo ""
