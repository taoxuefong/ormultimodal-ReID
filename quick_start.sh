#!/bin/bash

# PRCV Multi-Modal ReID 快速启动脚本

echo "=========================================="
echo "PRCV Multi-Modal ReID 快速启动"
echo "=========================================="

# 检查Python环境
echo "检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python 3.7+"
    exit 1
fi

# 检查PyTorch
echo "检查PyTorch..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" || {
    echo "错误: 未找到PyTorch，请先安装PyTorch"
    exit 1
}

# 检查CUDA
echo "检查CUDA..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')" || {
    echo "警告: CUDA不可用，将使用CPU训练（速度较慢）"
}

# 创建输出目录
echo "创建输出目录..."
mkdir -p ./outputs

# 检查数据集路径
echo "检查数据集路径..."
if [ ! -d "/data/taoxuefeng/PRCV" ]; then
    echo "警告: 数据集路径 /data/taoxuefeng/PRCV 不存在"
    echo "请确保数据集已正确放置，或修改脚本中的路径"
fi

echo ""
echo "=========================================="
echo "环境检查完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo ""
echo "1. 训练模型:"
echo "   python train_prcv_simple.py --config_file configs/prcv_config.yaml"
echo ""
echo "2. 测试模型:"
echo "   python test_prcv_simple.py --config_file configs/prcv_config.yaml --checkpoint ./outputs/checkpoint.pth"
echo ""
echo "3. 查看帮助:"
echo "   python train_prcv_simple.py --help"
echo "   python test_prcv_simple.py --help"
echo ""
echo "4. 运行代码测试:"
echo "   python test_code.py"
echo ""
echo "=========================================="
echo "开始训练？(y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "开始训练..."
    python train_prcv_simple.py --config_file configs/prcv_config.yaml
else
    echo "退出。您可以稍后手动运行训练命令。"
fi
