#!/usr/bin/env python3
"""
简单的测试脚本来查看调试输出
"""

import sys
import os
sys.path.append('.')

from datasets.build import build_dataloader
from configs.prcv_config import get_default_config

def test_debug():
    """测试调试输出"""
    print("开始测试...")
    
    # 获取配置
    args = get_default_config()
    args.root_dir = '/data/taoxuefeng/PRCV'
    args.dataset_name = 'PRCV'
    
    try:
        # 构建数据加载器
        print("构建数据加载器...")
        _, val_loader, _ = build_dataloader(args)
        
        if val_loader:
            print("获取第一个批次...")
            batch = next(iter(val_loader))
            print(f"批次类型: {type(batch)}")
            print(f"批次键: {list(batch.keys())}")
            
            # 检查第一个样本
            if isinstance(batch, dict):
                first_sample = {k: v[0] if hasattr(v, '__getitem__') else v for k, v in batch.items()}
                print(f"第一个样本的query_idx: {first_sample.get('query_idx', 'N/A')}")
                print(f"第一个样本的query_type: {first_sample.get('query_type', 'N/A')}")
                print(f"第一个样本的modalities: {first_sample.get('modalities', 'N/A')}")
        
        print("测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_debug()
