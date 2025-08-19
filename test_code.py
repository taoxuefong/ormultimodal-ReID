#!/usr/bin/env python3
"""
简单的代码测试脚本，验证多模态ReID代码的正确性
"""

import os
import sys
import torch
import numpy as np

def test_imports():
    """测试所有必要的导入"""
    print("Testing imports...")
    
    try:
        from datasets import build_dataloader
        print("✓ datasets.build_dataloader imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import datasets.build_dataloader: {e}")
        return False
    
    try:
        from model import build_model
        print("✓ model.build_model imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import model.build_model: {e}")
        return False
    
    try:
        from utils.multimodal_evaluator import MultiModalEvaluator
        print("✓ MultiModalEvaluator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MultiModalEvaluator: {e}")
        return False
    
    try:
        from processor.processor import do_inference
        print("✓ do_inference imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import do_inference: {e}")
        return False
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("\nTesting model creation...")
    
    try:
        # 创建模拟参数
        class MockArgs:
            def __init__(self):
                self.dataset_name = 'PRCV'
                self.pretrain_choice = 'ViT-B/32'
                self.img_size = [384, 128]
                self.stride_size = [32, 32]
                self.temperature = 0.07
                self.fusion_way = 'concat'
                self.loss_names = 'itc+id'
        
        args = MockArgs()
        
        # 测试模型构建
        from model import build_model
        model = build_model(args, num_classes=100)
        print("✓ Model created successfully")
        print(f"  Model type: {type(model).__name__}")
        
        # 测试前向传播
        batch_size = 2
        mock_batch = {
            'vis_images': torch.randn(batch_size, 3, 384, 128),
            'nir_images': torch.randn(batch_size, 3, 384, 128),
            'cp_images': torch.randn(batch_size, 3, 384, 128),
            'sk_images': torch.randn(batch_size, 3, 384, 128),
            'caption_ids': torch.randint(0, 1000, (batch_size, 77)),
            'pids': torch.randint(0, 100, (batch_size,))
        }
        
        with torch.no_grad():
            output = model(mock_batch)
            print("✓ Forward pass successful")
            print(f"  Output keys: {list(output.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_structure():
    """测试数据集结构"""
    print("\nTesting dataset structure...")
    
    try:
        # 检查PRCV数据集类
        from datasets.prcv import PRCV
        print("✓ PRCV dataset class imported successfully")
        
        # 检查多模态数据集类
        from datasets.bases import MultiModalDataset, MultiModalQueryDataset
        print("✓ MultiModalDataset and MultiModalQueryDataset imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluator():
    """测试评估器"""
    print("\nTesting evaluator...")
    
    try:
        from utils.multimodal_evaluator import MultiModalEvaluator
        
        # 创建模拟参数和数据加载器
        class MockArgs:
            def __init__(self):
                self.dataset_name = 'PRCV'
        
        class MockDataLoader:
            def __iter__(self):
                return iter([])
        
        args = MockArgs()
        gallery_loader = MockDataLoader()
        query_loader = MockDataLoader()
        
        evaluator = MultiModalEvaluator(args, gallery_loader, query_loader)
        print("✓ MultiModalEvaluator created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("Multi-Modal ReID Code Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_dataset_structure,
        test_evaluator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Code is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
