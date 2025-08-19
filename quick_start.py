#!/usr/bin/env python3
"""
PRCV Multi-Modal ReID 快速启动脚本
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """检查Python版本"""
    print("检查Python环境...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"错误: Python版本过低 ({version.major}.{version.minor})，需要Python 3.7+")
        return False
    print(f"✓ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def check_pytorch():
    """检查PyTorch"""
    print("检查PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA版本: {torch.version.cuda}")
        else:
            print("⚠ CUDA不可用，将使用CPU训练（速度较慢）")
        
        return True
    except ImportError:
        print("✗ 未找到PyTorch，请先安装PyTorch")
        return False

def check_dependencies():
    """检查其他依赖"""
    print("检查其他依赖...")
    
    required_packages = [
        'numpy', 'torchvision', 'PIL', 'prettytable'
    ]
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✓ {package} (Pillow) 已安装")
            else:
                importlib.import_module(package)
                print(f"✓ {package} 已安装")
        except ImportError:
            print(f"⚠ {package} 未安装，可能需要安装")
    
    return True

def create_output_dirs():
    """创建输出目录"""
    print("创建输出目录...")
    dirs = ['./outputs', './configs']
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")

def check_dataset_path():
    """检查数据集路径"""
    print("检查数据集路径...")
    dataset_path = "/data/taoxuefeng/PRCV"
    
    if os.path.exists(dataset_path):
        print(f"✓ 数据集路径存在: {dataset_path}")
        return True
    else:
        print(f"⚠ 数据集路径不存在: {dataset_path}")
        print("  请确保数据集已正确放置，或修改脚本中的路径")
        return False

def run_command(cmd, description):
    """运行命令"""
    print(f"\n{description}...")
    print(f"命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ 命令执行成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 命令执行失败: {e}")
        if e.stdout:
            print(f"输出: {e.stdout}")
        if e.stderr:
            print(f"错误: {e.stderr}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("PRCV Multi-Modal ReID 快速启动")
    print("=" * 50)
    
    # 环境检查
    if not check_python_version():
        return False
    
    if not check_pytorch():
        return False
    
    check_dependencies()
    create_output_dirs()
    check_dataset_path()
    
    print("\n" + "=" * 50)
    print("环境检查完成！")
    print("=" * 50)
    
    # 显示使用方法
    print("\n使用方法:")
    print("\n1. 训练模型:")
    print("   python train_prcv_simple.py --config_file configs/prcv_config.yaml")
    print("\n2. 测试模型:")
    print("   python test_prcv_simple.py --config_file configs/prcv_config.yaml --checkpoint ./outputs/checkpoint.pth")
    print("\n3. 查看帮助:")
    print("   python train_prcv_simple.py --help")
    print("   python test_prcv_simple.py --help")
    print("\n4. 运行代码测试:")
    print("   python test_code.py")
    
    # 询问是否开始训练
    print("\n" + "=" * 50)
    response = input("开始训练？(y/n): ").lower().strip()
    
    if response in ['y', 'yes', '是']:
        print("开始训练...")
        
        # 检查配置文件是否存在
        config_file = "configs/prcv_config.yaml"
        if not os.path.exists(config_file):
            print(f"配置文件 {config_file} 不存在，请先创建配置文件")
            return False
        
        # 运行训练
        cmd = f"python train_prcv_simple.py --config_file {config_file}"
        return run_command(cmd, "开始训练")
    else:
        print("退出。您可以稍后手动运行训练命令。")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
