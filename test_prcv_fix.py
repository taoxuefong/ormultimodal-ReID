#!/usr/bin/env python3
"""
测试PRCV数据集修复后的功能
"""

import sys
import os

def test_prcv_dataset():
    """测试PRCV数据集"""
    print("测试PRCV数据集...")
    
    try:
        # 添加当前目录到Python路径
        sys.path.insert(0, os.getcwd())
        
        # 测试导入
        from datasets.prcv import PRCV
        print("✅ PRCV数据集类导入成功")
        
        # 测试创建实例
        dataset = PRCV(root="/data/taoxuefeng/PRCV")
        print("✅ PRCV数据集实例创建成功")
        
        # 测试训练数据
        print(f"训练样本数: {len(dataset.train)}")
        if len(dataset.train) > 0:
            sample = dataset.train[0]
            print(f"第一个样本: {sample}")
            
            # 检查路径
            pid, image_id, vis_path, nir_path, cp_path, sk_path, caption = sample
            print(f"  PID: {pid}")
            print(f"  图像ID: {image_id}")
            print(f"  RGB路径: {vis_path}")
            print(f"  描述: {caption[:50]}...")
            
            # 检查文件是否存在
            if os.path.exists(vis_path):
                print(f"  ✅ RGB图像文件存在")
            else:
                print(f"  ❌ RGB图像文件不存在: {vis_path}")
                # 检查父目录
                parent_dir = os.path.dirname(vis_path)
                if os.path.exists(parent_dir):
                    print(f"  ✅ 父目录存在: {parent_dir}")
                    files = os.listdir(parent_dir)
                    print(f"  父目录中的文件: {files[:5]}...")
                else:
                    print(f"  ❌ 父目录不存在: {parent_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader():
    """测试数据加载器"""
    print("\n测试数据加载器...")
    
    try:
        from datasets import build_dataloader
        
        # 创建模拟参数
        class MockArgs:
            def __init__(self):
                self.dataset_name = 'PRCV'
                self.root_dir = '/data/taoxuefeng/PRCV'
                self.training = True
                self.batch_size = 2  # 使用小批次进行测试
                self.num_workers = 0  # 使用单进程避免多进程问题
                self.img_size = [384, 128]
                self.img_aug = False  # 关闭数据增强
                self.val_dataset = 'val'
                self.sampler = 'random'  # 使用随机采样器
                self.num_instance = 2
                self.text_length = 77
                self.nlp_aug = False
        
        args = MockArgs()
        
        # 测试构建数据加载器
        train_loader, val_img_loader, val_txt_loader, val_sketch_loader, num_classes = build_dataloader(args)
        print("✅ 数据加载器构建成功")
        print(f"训练集类别数: {num_classes}")
        
        # 测试加载一个批次
        print("测试加载一个批次...")
        for batch in train_loader:
            print(f"批次大小: {batch['vis_images'].shape}")
            print(f"批次键: {list(batch.keys())}")
            print(f"PID: {batch['pids']}")
            print(f"图像ID: {batch['image_ids']}")
            break  # 只测试第一个批次
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("PRCV数据集修复测试")
    print("=" * 50)
    
    # 测试PRCV数据集
    dataset_ok = test_prcv_dataset()
    
    if dataset_ok:
        # 测试数据加载器
        loader_ok = test_dataloader()
        
        if loader_ok:
            print("\n🎉 所有测试通过！数据集修复成功。")
            return True
        else:
            print("\n❌ 数据加载器测试失败。")
            return False
    else:
        print("\n❌ PRCV数据集测试失败。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
