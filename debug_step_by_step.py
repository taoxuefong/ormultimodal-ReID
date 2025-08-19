#!/usr/bin/env python3
"""
逐步调试脚本
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
from datasets.build import build_dataloader
from model import build_model
from solver import build_optimizer, build_lr_scheduler
from utils.options import get_args

def debug_step_by_step():
    """逐步调试每个组件"""
    print("=== Step-by-Step Debug ===")
    
    # 步骤1: 获取参数
    print("\n1. Getting arguments...")
    try:
        args = get_args()
        args.dataset_name = 'PRCV'
        args.root_dir = '/data/taoxuefeng/PRCV'
        args.training = True
        args.batch_size = 32
        args.num_epoch = 100
        args.lr = 0.0001
        args.sampler = 'balanced'
        args.num_instance = 4
        print("✅ Arguments loaded successfully")
    except Exception as e:
        print(f"❌ Error loading arguments: {e}")
        return
    
    # 步骤2: 构建数据加载器
    print("\n2. Building dataloader...")
    try:
        train_loader, val_img_loader, val_txt_loader, val_sketch_loader, num_classes = build_dataloader(args)
        print(f"✅ DataLoader built successfully")
        print(f"   Train loader: {len(train_loader)} batches")
        print(f"   Batch size: {train_loader.batch_size}")
    except Exception as e:
        print(f"❌ Error building dataloader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤3: 测试第一个batch
    print("\n3. Testing first batch...")
    try:
        print("   Getting first batch...")
        first_batch = next(iter(train_loader))
        print(f"✅ First batch loaded successfully")
        print(f"   Batch keys: {list(first_batch.keys())}")
        
        if 'vis_images' in first_batch:
            print(f"   Images shape: {first_batch['vis_images'].shape}")
        if 'pids' in first_batch:
            print(f"   PIDs shape: {first_batch['pids'].shape}")
            print(f"   PIDs: {first_batch['pids']}")
    except Exception as e:
        print(f"❌ Error loading first batch: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤4: 构建模型
    print("\n4. Building model...")
    try:
        model = build_model(args, num_classes)
        print(f"✅ Model built successfully")
        print(f"   Model type: {type(model)}")
    except Exception as e:
        print(f"❌ Error building model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤5: 移动模型到GPU
    print("\n5. Moving model to GPU...")
    try:
        device = "cuda"
        model = model.to(device)
        print(f"✅ Model moved to GPU successfully")
    except Exception as e:
        print(f"❌ Error moving model to GPU: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤6: 构建优化器和调度器
    print("\n6. Building optimizer and scheduler...")
    try:
        optimizer = build_optimizer(args, model)
        scheduler = build_lr_scheduler(args, optimizer)
        print(f"✅ Optimizer and scheduler built successfully")
    except Exception as e:
        print(f"❌ Error building optimizer/scheduler: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤7: 测试模型前向传播
    print("\n7. Testing model forward pass...")
    try:
        print("   Moving batch to GPU...")
        gpu_batch = {k: v.to(device) for k, v in first_batch.items()}
        print("   ✅ Batch moved to GPU")
        
        print("   Running model forward pass...")
        model.train()
        with torch.no_grad():  # 不计算梯度，只测试前向传播
            ret = model(gpu_batch)
        print(f"✅ Model forward pass completed successfully")
        print(f"   Return keys: {list(ret.keys())}")
        
        if 'total_loss' in ret:
            print(f"   Total loss: {ret['total_loss']}")
        else:
            print(f"   Loss keys: {[k for k in ret.keys() if 'loss' in k]}")
            
    except Exception as e:
        print(f"❌ Error in model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 All steps completed successfully!")
    print("The issue might be in the training loop or data iteration.")

if __name__ == "__main__":
    debug_step_by_step()
