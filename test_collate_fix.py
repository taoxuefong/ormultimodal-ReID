#!/usr/bin/env python3
"""
测试collate修复的脚本
"""

import sys
import os

def test_collate_fix():
    """测试collate修复"""
    print("测试collate修复...")
    
    try:
        # 添加当前目录到Python路径
        sys.path.insert(0, os.getcwd())
        
        # 测试导入
        from datasets.build import multimodal_collate
        
        print("✅ multimodal_collate函数导入成功")
        
        # 创建模拟的batch数据
        mock_batch = [
            {
                'query_idx': 0,
                'query_type': 'onemodal_NIR',
                'modalities': ['nir'],
                'nir_images': None,  # 模拟None值
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
        
        # 测试collate函数
        try:
            result = multimodal_collate(mock_batch)
            print("✅ multimodal_collate函数执行成功")
            print(f"结果键: {list(result.keys())}")
            
            # 检查结果
            for key, value in result.items():
                print(f"  {key}: 类型={type(value)}, 值={value}")
            
            return True
            
        except Exception as e:
            print(f"❌ multimodal_collate函数执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ collate修复测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("Collate修复测试")
    print("=" * 50)
    
    success = test_collate_fix()
    
    if success:
        print("\n🎉 Collate修复测试通过！")
        return True
    else:
        print("\n❌ Collate修复测试失败，请检查错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
