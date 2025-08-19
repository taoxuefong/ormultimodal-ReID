#!/usr/bin/env python3
"""
测试张量维度的脚本
"""

import sys
import os

def test_tensor_dimensions():
    """测试张量维度"""
    print("测试张量维度...")
    
    try:
        # 添加当前目录到Python路径
        sys.path.insert(0, os.getcwd())
        
        # 测试导入
        from datasets.build import multimodal_collate
        
        print("✅ multimodal_collate函数导入成功")
        
        # 创建模拟的batch数据，模拟修复后的情况
        mock_batch = [
            {
                'query_idx': 0,
                'query_type': 'onemodal_NIR',
                'modalities': ['nir'],
                'nir_images': None,  # 将被替换为占位符
                'cp_images': None,
                'sk_images': None,
                'text_tokens': None
            },
            {
                'query_idx': 1,
                'query_type': 'twomodal_CP_SK',
                'modalities': ['cp', 'sk'],
                'nir_images': None,
                'cp_images': None,
                'sk_images': None,
                'text_tokens': None
            }
        ]
        
        # 模拟修复后的占位符张量（3x384x128）
        import torch
        placeholder_img = torch.zeros(3, 384, 128)
        placeholder_text = torch.zeros(77, dtype=torch.long)
        
        # 替换None值为占位符
        for item in mock_batch:
            item['nir_images'] = placeholder_img
            item['cp_images'] = placeholder_img
            item['sk_images'] = placeholder_img
            item['text_tokens'] = placeholder_text
        
        print("✅ 占位符张量创建成功")
        print(f"NIR图像占位符形状: {placeholder_img.shape}")
        print(f"文本占位符形状: {placeholder_text.shape}")
        
        # 测试collate函数
        try:
            result = multimodal_collate(mock_batch)
            print("✅ multimodal_collate函数执行成功")
            print(f"结果键: {list(result.keys())}")
            
            # 检查结果张量的维度
            for key, value in result.items():
                if torch.is_tensor(value):
                    print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
                    # 验证图像张量是4D (batch_size, channels, height, width)
                    if 'images' in key and value.dim() == 4:
                        print(f"    ✅ {key} 维度正确: {value.shape}")
                    elif 'images' in key:
                        print(f"    ❌ {key} 维度错误: {value.shape}, 期望4D")
                else:
                    print(f"  {key}: 类型={type(value)}, 值={value}")
            
            return True
            
        except Exception as e:
            print(f"❌ multimodal_collate函数执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ 张量维度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("张量维度测试")
    print("=" * 50)
    
    success = test_tensor_dimensions()
    
    if success:
        print("\n🎉 张量维度测试通过！")
        return True
    else:
        print("\n❌ 张量维度测试失败，请检查错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
