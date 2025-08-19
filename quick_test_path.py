#!/usr/bin/env python3
"""
快速测试路径的脚本
"""

import os

def test_paths():
    """测试路径"""
    print("测试PRCV数据集路径...")
    
    base_dir = '/data/taoxuefeng/PRCV'
    val_dir = os.path.join(base_dir, 'val')
    
    print(f"基础目录: {base_dir}")
    print(f"验证目录: {val_dir}")
    
    # 检查目录是否存在
    if os.path.exists(base_dir):
        print(f"✅ 基础目录存在: {base_dir}")
    else:
        print(f"❌ 基础目录不存在: {base_dir}")
        return False
    
    if os.path.exists(val_dir):
        print(f"✅ 验证目录存在: {val_dir}")
    else:
        print(f"❌ 验证目录不存在: {val_dir}")
        return False
    
    # 检查子目录
    subdirs = ['nir', 'cp', 'sk', 'gallery']
    for subdir in subdirs:
        subdir_path = os.path.join(val_dir, subdir)
        if os.path.exists(subdir_path):
            print(f"✅ 子目录存在: {subdir_path}")
            # 列出前几个文件
            try:
                files = os.listdir(subdir_path)[:3]
                print(f"  示例文件: {files}")
            except Exception as e:
                print(f"  无法列出文件: {e}")
        else:
            print(f"❌ 子目录不存在: {subdir_path}")
    
    # 测试具体文件路径
    test_files = [
        'nir/4712.jpg',
        'cp/2522.jpg',
        'sk/3019.jpg'
    ]
    
    print("\n测试具体文件路径:")
    for file_path in test_files:
        full_path = os.path.join(val_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ 文件存在: {full_path}")
        else:
            print(f"❌ 文件不存在: {full_path}")
    
    return True

if __name__ == "__main__":
    success = test_paths()
    if success:
        print("\n🎉 路径测试完成！")
    else:
        print("\n❌ 路径测试失败！")
