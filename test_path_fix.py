#!/usr/bin/env python3
"""
测试路径修复的脚本
"""

import sys
import os
import torch

def test_path_fix():
    """测试路径修复"""
    print("测试路径修复...")
    
    try:
        # 添加当前目录到Python路径
        sys.path.insert(0, os.getcwd())
        
        # 测试导入
        from datasets.bases import MultiModalQueryDataset
        from torchvision import transforms
        
        print("✅ MultiModalQueryDataset导入成功")
        
        # 创建简单的transform
        transform = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建模拟的查询数据
        mock_queries = [
            {
                'query_idx': 0,
                'query_type': 'onemodal_NIR',
                'modalities': ['nir'],
                'content': ['nir/4712.jpg']
            },
            {
                'query_idx': 1,
                'query_type': 'twomodal_CP_SK',
                'modalities': ['cp', 'sk'],
                'content': ['cp/2522.jpg', 'sk/3019.jpg']
            }
        ]
        
        # 创建数据集实例，指定基础目录
        base_dir = '/data/taoxuefeng/PRCV/val'
        dataset = MultiModalQueryDataset(mock_queries, transform, base_dir=base_dir)
        
        print(f"✅ MultiModalQueryDataset创建成功，基础目录: {base_dir}")
        print(f"数据集大小: {len(dataset)}")
        
        # 测试第一个样本
        if len(dataset) > 0:
            print("\n测试第一个样本...")
            try:
                sample = dataset[0]
                print(f"样本类型: {type(sample)}")
                if isinstance(sample, dict):
                    print(f"样本键: {list(sample.keys())}")
                    for key, value in sample.items():
                        if torch.is_tensor(value):
                            print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
                        else:
                            print(f"  {key}: 类型={type(value)}, 值={value}")
                else:
                    print(f"样本内容: {sample}")
                print("✅ 第一个样本加载成功")
            except Exception as e:
                print(f"❌ 第一个样本加载失败: {e}")
                return False
        
        # 测试第二个样本
        if len(dataset) > 1:
            print("\n测试第二个样本...")
            try:
                sample = dataset[1]
                print(f"样本类型: {type(sample)}")
                if isinstance(sample, dict):
                    print(f"样本键: {list(sample.keys())}")
                    for key, value in sample.items():
                        if torch.is_tensor(value):
                            print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
                        else:
                            print(f"  {key}: 类型={type(value)}, 值={value}")
                else:
                    print(f"样本内容: {sample}")
                print("✅ 第二个样本加载成功")
            except Exception as e:
                print(f"❌ 第二个样本加载失败: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 路径修复测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_existence():
    """测试文件是否存在"""
    print("\n测试文件存在性...")
    
    base_dir = '/data/taoxuefeng/PRCV/val'
    
    test_files = [
        'nir/4712.jpg',
        'cp/2522.jpg',
        'sk/3019.jpg'
    ]
    
    for file_path in test_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ 文件存在: {full_path}")
        else:
            print(f"❌ 文件不存在: {full_path}")
    
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("路径修复测试")
    print("=" * 50)
    
    # 测试文件存在性
    file_ok = test_file_existence()
    
    # 测试路径修复
    path_ok = test_path_fix()
    
    if file_ok and path_ok:
        print("\n🎉 所有测试通过！路径修复成功。")
        return True
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
