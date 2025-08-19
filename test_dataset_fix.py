#!/usr/bin/env python3
"""
测试PRCV数据集修复后的功能
"""

import sys
import os

def test_dataset_attributes():
    """测试数据集属性"""
    print("测试PRCV数据集属性...")
    
    try:
        # 添加当前目录到Python路径
        sys.path.insert(0, os.getcwd())
        
        # 测试导入
        from datasets.prcv import PRCV
        print("✅ PRCV数据集类导入成功")
        
        # 测试创建实例
        dataset = PRCV(root="/data/taoxuefeng/PRCV")
        print("✅ PRCV数据集实例创建成功")
        
        # 测试必要属性
        required_attrs = ['train', 'val', 'test', 'train_id_container', 'val_id_container', 'test_id_container']
        for attr in required_attrs:
            if hasattr(dataset, attr):
                print(f"✅ 属性 {attr} 存在")
            else:
                print(f"❌ 属性 {attr} 不存在")
                return False
        
        # 测试数据集大小
        print(f"训练集大小: {len(dataset.train)}")
        print(f"验证集查询数: {len(dataset.val['queries'])}")
        print(f"测试集查询数: {len(dataset.test['queries'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader_build():
    """测试数据加载器构建"""
    print("\n测试数据加载器构建...")
    
    try:
        from datasets import build_dataloader
        
        # 创建模拟参数
        class MockArgs:
            def __init__(self):
                self.dataset_name = 'PRCV'
                self.root_dir = '/data/taoxuefeng/PRCV'
                self.training = True
                self.batch_size = 32
                self.num_workers = 4
                self.img_size = [384, 128]
                self.img_aug = True
                self.val_dataset = 'val'
                self.sampler = 'identity'
                self.num_instance = 4
                self.text_length = 77
                self.nlp_aug = False
        
        args = MockArgs()
        
        # 测试构建数据加载器
        train_loader, val_img_loader, val_txt_loader, val_sketch_loader, num_classes = build_dataloader(args)
        print("✅ 数据加载器构建成功")
        print(f"训练集类别数: {num_classes}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("PRCV数据集修复测试")
    print("=" * 50)
    
    # 测试数据集属性
    attrs_ok = test_dataset_attributes()
    
    if attrs_ok:
        # 测试数据加载器构建
        loader_ok = test_dataloader_build()
        
        if loader_ok:
            print("\n🎉 所有测试通过！数据集修复成功。")
            return True
        else:
            print("\n❌ 数据加载器构建测试失败。")
            return False
    else:
        print("\n❌ 数据集属性测试失败。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
