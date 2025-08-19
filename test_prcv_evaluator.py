#!/usr/bin/env python3
"""
测试PRCV评估器的脚本
"""

import sys
import os
import torch

def test_prcv_evaluator():
    """测试PRCV评估器"""
    print("测试PRCV评估器...")
    
    try:
        # 添加当前目录到Python路径
        sys.path.insert(0, os.getcwd())
        
        # 测试导入
        from utils.prcv_evaluator import PRCVEvaluator
        print("✅ PRCV评估器导入成功")
        
        # 创建模拟参数和数据加载器
        class MockArgs:
            def __init__(self):
                self.dataset_name = 'PRCV'
        
        class MockDataLoader:
            def __iter__(self):
                # 返回模拟的批次数据
                batch = [
                    torch.randn(2, 3, 384, 128),  # 图像
                    torch.tensor([1, 2]),           # PIDs
                    torch.tensor([0, 0])            # Camera IDs
                ]
                yield batch
        
        args = MockArgs()
        gallery_loader = MockDataLoader()
        query_loader = MockDataLoader()
        
        # 创建评估器
        evaluator = PRCVEvaluator(args, gallery_loader, query_loader)
        print("✅ PRCV评估器创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ PRCV评估器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_format_handling():
    """测试数据格式处理"""
    print("\n测试数据格式处理...")
    
    try:
        from utils.prcv_evaluator import PRCVEvaluator
        
        class MockArgs:
            def __init__(self):
                self.dataset_name = 'PRCV'
        
        # 测试列表格式的batch
        class ListDataLoader:
            def __iter__(self):
                batch = [
                    torch.randn(2, 3, 384, 128),  # 图像
                    torch.tensor([1, 2]),           # PIDs
                    torch.tensor([0, 0])            # Camera IDs
                ]
                yield batch
        
        # 测试字典格式的batch
        class DictDataLoader:
            def __iter__(self):
                batch = {
                    'vis_images': torch.randn(2, 3, 384, 128),
                    'image_pids': torch.tensor([1, 2]),
                    'image_camids': torch.tensor([0, 0])
                }
                yield batch
        
        args = MockArgs()
        
        # 测试列表格式
        list_evaluator = PRCVEvaluator(args, ListDataLoader(), ListDataLoader())
        print("✅ 列表格式数据加载器处理成功")
        
        # 测试字典格式
        dict_evaluator = PRCVEvaluator(args, DictDataLoader(), DictDataLoader())
        print("✅ 字典格式数据加载器处理成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据格式处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("PRCV评估器测试")
    print("=" * 50)
    
    # 测试PRCV评估器
    evaluator_ok = test_prcv_evaluator()
    
    if evaluator_ok:
        # 测试数据格式处理
        format_ok = test_data_format_handling()
        
        if format_ok:
            print("\n🎉 所有测试通过！PRCV评估器准备就绪。")
            return True
        else:
            print("\n❌ 数据格式处理测试失败。")
            return False
    else:
        print("\n❌ PRCV评估器测试失败。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
