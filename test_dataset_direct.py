#!/usr/bin/env python3
"""
直接测试MultiModalQueryDataset的脚本
"""

import sys
import os
sys.path.append('.')

# 模拟环境
import torch
import torchvision.transforms as T
from datasets.bases import MultiModalQueryDataset
from utils.tokenizer import SimpleTokenizer

def test_dataset_direct():
    """直接测试数据集类"""
    print("开始直接测试...")
    
    # 模拟一个查询
    query = {
        'query_idx': 60717,
        'query_type': 'threemodal_TEXT_NIR_CP',
        'content': [
            "A tall, slim young man with thick, short black hair and a pair of glasses framed in dark rims. He is wearing a long-sleeved black jacket over a gray T-shirt, and on the lower half, he dons black casual trousers that fall to his ankles. On his feet are a pair of light gray casual shoes with white soles.",
            "nir/249.jpg",
            "cp/2295.jpg"
        ]
    }
    
    # 模拟数据集
    queries = [query]
    
    # 创建数据集实例
    base_dir = '/data/taoxuefeng/PRCV/val'
    transform = T.Compose([
        T.Resize((384, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MultiModalQueryDataset(
        queries=queries,
        transform=transform,
        text_length=77,
        truncate=True,
        base_dir=base_dir
    )
    
    print(f"数据集长度: {len(dataset)}")
    
    try:
        # 获取第一个样本
        print("获取第一个样本...")
        sample = dataset[0]
        
        print(f"样本键: {list(sample.keys())}")
        print(f"query_idx: {sample['query_idx']}")
        print(f"query_type: {sample['query_type']}")
        print(f"modalities: {sample['modalities']}")
        
        # 检查各个模态
        if 'nir_images' in sample:
            print(f"nir_images 形状: {sample['nir_images'].shape}")
        if 'cp_images' in sample:
            print(f"cp_images 形状: {sample['cp_images'].shape}")
        if 'sk_images' in sample:
            print(f"sk_images 形状: {sample['sk_images'].shape}")
        if 'text_tokens' in sample:
            print(f"text_tokens 形状: {sample['text_tokens'].shape}")
        
        print("✅ 测试成功！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_dataset_direct()
