#!/usr/bin/env python3
"""
测试平衡采样器
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from datasets.balanced_sampler import PRCVBalancedSampler
from datasets.build import build_dataloader
from utils.options import get_args

def test_balanced_sampler():
    """测试平衡采样器"""
    print("Testing Balanced Sampler...")
    
    # 获取参数
    args = get_args()
    args.dataset_name = 'PRCV'
    args.root_dir = '/data/taoxuefeng/PRCV'
    args.training = True
    args.batch_size = 32
    args.num_epoch = 100
    args.lr = 0.0001
    args.sampler = 'balanced'
    args.num_instance = 4
    
    print(f"Args: {args}")
    
    try:
        # 构建数据加载器
        print("\nBuilding dataloader...")
        train_loader, val_img_loader, val_txt_loader, val_sketch_loader, num_classes = build_dataloader(args)
        
        print(f"Data loader built successfully!")
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Batch size: {train_loader.batch_size}")
        
        # 测试几个batch
        print("\nTesting first few batches...")
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # 只测试前3个batch
                break
                
            print(f"\nBatch {batch_idx}:")
            print(f"  Batch keys: {list(batch.keys())}")
            
            if 'pids' in batch:
                pids = batch['pids']
                unique_pids = torch.unique(pids)
                print(f"  PIDs shape: {pids.shape}")
                print(f"  Unique PIDs: {unique_pids.tolist()}")
                print(f"  Num unique PIDs: {len(unique_pids)}")
                
                # 检查每个类别的样本数量
                for pid in unique_pids:
                    count = (pids == pid).sum().item()
                    print(f"    PID {pid}: {count} samples")
            
            if 'vis_images' in batch:
                print(f"  Images shape: {batch['vis_images'].shape}")
            
            if 'text' in batch:
                print(f"  Text shape: {batch['text'].shape}")
        
        print("\nBalanced sampler test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_balanced_sampler()
