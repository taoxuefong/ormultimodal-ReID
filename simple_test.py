#!/usr/bin/env python3
"""
简单测试平衡采样器
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from datasets.balanced_sampler import PRCVBalancedSampler

def test_sampler():
    """测试采样器"""
    print("Testing PRCVBalancedSampler...")
    
    # 创建一个模拟的PRCV数据集
    # 格式: (pid, image_id, vis_path, nir_path, cp_path, sk_path, caption)
    mock_dataset = []
    for pid in range(10):  # 10个类别
        for i in range(5):  # 每个类别5个样本
            mock_dataset.append((
                pid,  # pid
                i,    # image_id
                f"vis/{pid:04d}/sample_{i}.jpg",  # vis_path
                f"nir/{pid:04d}/sample_{i}.jpg",  # nir_path
                f"cp/{pid:04d}/sample_{i}.jpg",   # cp_path
                f"sk/{pid:04d}/sample_{i}.jpg",   # sk_path
                f"person {pid} sample {i}"        # caption
            ))
    
    print(f"Mock dataset: {len(mock_dataset)} samples, {len(set(item[0] for item in mock_dataset))} classes")
    
    try:
        # 创建平衡采样器
        sampler = PRCVBalancedSampler(
            dataset=mock_dataset,
            n_classes_per_batch=4,  # 每个batch 4个类别
            n_samples_per_class=2   # 每个类别2个样本
        )
        
        print(f"Sampler created successfully!")
        print(f"Batches per epoch: {len(sampler)}")
        
        # 测试第一个batch
        print("\nTesting first batch...")
        for batch_idx, batch_indices in enumerate(sampler):
            if batch_idx >= 2:  # 只测试前2个batch
                break
                
            print(f"\nBatch {batch_idx}:")
            print(f"  Indices: {batch_indices}")
            
            # 分析这个batch的类别分布
            batch_pids = [mock_dataset[idx][0] for idx in batch_indices]
            unique_pids = list(set(batch_pids))
            print(f"  Unique PIDs: {unique_pids}")
            
            for pid in unique_pids:
                count = batch_pids.count(pid)
                print(f"    PID {pid}: {count} samples")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sampler()
