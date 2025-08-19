#!/usr/bin/env python3
"""
测试PRCV训练流程的脚本
"""

import sys
import os
import torch

def test_model_forward():
    """测试模型前向传播"""
    print("测试模型前向传播...")
    
    try:
        # 添加当前目录到Python路径
        sys.path.insert(0, os.getcwd())
        
        # 测试导入
        from model import build_model
        
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
        
        # 构建模型
        model = build_model(args, num_classes=100)
        print("✅ 模型构建成功")
        
        # 创建模拟批次数据
        batch_size = 2
        mock_batch = {
            'vis_images': torch.randn(batch_size, 3, 384, 128),
            'nir_images': torch.randn(batch_size, 3, 384, 128),
            'cp_images': torch.randn(batch_size, 3, 384, 128),
            'sk_images': torch.randn(batch_size, 3, 384, 128),
            'caption_ids': torch.randint(0, 1000, (batch_size, 77)),
            'pids': torch.randint(0, 100, (batch_size,))
        }
        
        # 测试前向传播
        model.train()
        with torch.no_grad():
            output = model(mock_batch)
            print("✅ 前向传播成功")
            print(f"输出键: {list(output.keys())}")
            
            # 检查损失
            if 'itc_loss' in output:
                print(f"ITC损失: {output['itc_loss'].item():.4f}")
            if 'nir_loss' in output:
                print(f"NIR损失: {output['nir_loss'].item():.4f}")
            if 'cp_loss' in output:
                print(f"CP损失: {output['cp_loss'].item():.4f}")
            if 'sk_loss' in output:
                print(f"SK损失: {output['sk_loss'].item():.4f}")
            if 'text_loss' in output:
                print(f"文本损失: {output['text_loss'].item():.4f}")
            if 'temperature' in output:
                print(f"温度: {output['temperature'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader_batch():
    """测试数据加载器的批次数据"""
    print("\n测试数据加载器批次数据...")
    
    try:
        from datasets import build_dataloader
        
        # 创建模拟参数
        class MockArgs:
            def __init__(self):
                self.dataset_name = 'PRCV'
                self.root_dir = '/data/taoxuefeng/PRCV'
                self.training = True
                self.batch_size = 2
                self.num_workers = 0  # 使用单进程
                self.img_size = [384, 128]
                self.img_aug = False
                self.val_dataset = 'val'
                self.sampler = 'random'
                self.num_instance = 2
                self.text_length = 77
                self.nlp_aug = False
        
        args = MockArgs()
        
        # 构建数据加载器
        train_loader, val_img_loader, val_txt_loader, val_sketch_loader, num_classes = build_dataloader(args)
        print("✅ 数据加载器构建成功")
        
        # 测试一个批次
        for batch in train_loader:
            print(f"批次键: {list(batch.keys())}")
            print(f"批次大小: {batch['vis_images'].shape}")
            print(f"PID: {batch['pids']}")
            print(f"图像ID: {batch['image_ids']}")
            print(f"描述ID形状: {batch['caption_ids'].shape}")
            
            # 检查数据类型
            print(f"数据类型:")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {type(value)} - {value.shape} - {value.dtype}")
                else:
                    print(f"  {key}: {type(value)} - {value}")
            
            break  # 只测试第一个批次
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器批次测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_compatibility():
    """测试训练兼容性"""
    print("\n测试训练兼容性...")
    
    try:
        # 模拟训练循环中的批次处理
        batch = {
            'vis_images': torch.randn(2, 3, 384, 128),
            'nir_images': torch.randn(2, 3, 384, 128),
            'cp_images': torch.randn(2, 3, 384, 128),
            'sk_images': torch.randn(2, 3, 384, 128),
            'caption_ids': torch.randint(0, 1000, (2, 77)),
            'pids': torch.randint(0, 100, (2,))
        }
        
        # 测试批次大小获取
        batch_size = batch.get('images', batch.get('vis_images', torch.tensor(1))).shape[0]
        print(f"✅ 批次大小获取成功: {batch_size}")
        
        # 测试模型前向传播
        from model import build_model
        
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
        model = build_model(args, num_classes=100)
        
        with torch.no_grad():
            output = model(batch)
            print("✅ 训练兼容性测试成功")
            print(f"输出键: {list(output.keys())}")
            
            # 检查是否有损失
            losses = [v for k, v in output.items() if "loss" in k]
            if losses:
                total_loss = sum(losses)
                print(f"总损失: {total_loss.item():.4f}")
            else:
                print("⚠ 没有找到损失项")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("PRCV训练流程测试")
    print("=" * 50)
    
    # 测试模型前向传播
    model_ok = test_model_forward()
    
    if model_ok:
        # 测试数据加载器批次
        loader_ok = test_dataloader_batch()
        
        if loader_ok:
            # 测试训练兼容性
            compat_ok = test_training_compatibility()
            
            if compat_ok:
                print("\n🎉 所有测试通过！训练流程准备就绪。")
                return True
            else:
                print("\n❌ 训练兼容性测试失败。")
                return False
        else:
            print("\n❌ 数据加载器批次测试失败。")
            return False
    else:
        print("\n❌ 模型前向传播测试失败。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
