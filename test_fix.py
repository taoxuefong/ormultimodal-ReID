#!/usr/bin/env python3
"""
测试MultiModalQueryDataset修复的脚本
"""

import sys
import os
sys.path.append('.')

from datasets.build import build_dataloader
from configs.prcv_config import get_default_config

def test_query_dataset():
    """测试查询数据集是否能正确加载"""
    print("开始测试查询数据集...")
    
    # 获取配置
    args = get_default_config()
    args.root_dir = '/data/taoxuefeng/PRCV'
    args.dataset_name = 'PRCV'
    
    try:
        # 构建数据加载器
        print("构建数据加载器...")
        _, val_loader, test_loader = build_dataloader(args)
        
        print(f"验证集加载器类型: {type(val_loader)}")
        print(f"测试集加载器类型: {type(test_loader)}")
        
        # 测试验证集
        if val_loader:
            print("\n测试验证集...")
            batch = next(iter(val_loader))
            print(f"批次类型: {type(batch)}")
            print(f"批次键: {list(batch.keys())}")
            
            # 检查第一个样本
            if isinstance(batch, dict):
                first_sample = {k: v[0] if hasattr(v, '__getitem__') else v for k, v in batch.items()}
                print(f"第一个样本的query_idx: {first_sample.get('query_idx', 'N/A')}")
                print(f"第一个样本的query_type: {first_sample.get('query_type', 'N/A')}")
                print(f"第一个样本的modalities: {first_sample.get('modalities', 'N/A')}")
                
                # 检查图像张量
                for key in ['nir_images', 'cp_images', 'sk_images']:
                    if key in first_sample:
                        tensor = first_sample[key]
                        if hasattr(tensor, 'shape'):
                            print(f"{key} 形状: {tensor.shape}")
                        else:
                            print(f"{key} 类型: {type(tensor)}")
                
                # 检查文本token
                if 'text_tokens' in first_sample:
                    text_tokens = first_sample['text_tokens']
                    if hasattr(text_tokens, 'shape'):
                        print(f"text_tokens 形状: {text_tokens.shape}")
                    else:
                        print(f"text_tokens 类型: {type(text_tokens)}")
        
        print("\n✅ 查询数据集测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_query_dataset()
