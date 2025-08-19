#!/usr/bin/env python3
"""
测试PRCV数据集路径和结构的脚本
"""

import os
import os.path as op

def test_dataset_structure():
    """测试数据集结构"""
    print("=" * 50)
    print("PRCV数据集路径和结构测试")
    print("=" * 50)
    
    # 测试路径
    base_path = "/data/taoxuefeng/PRCV"
    print(f"基础路径: {base_path}")
    
    if not os.path.exists(base_path):
        print(f"❌ 基础路径不存在: {base_path}")
        return False
    
    print(f"✅ 基础路径存在: {base_path}")
    
    # 测试训练目录
    train_path = op.join(base_path, "train")
    print(f"\n训练目录: {train_path}")
    
    if not os.path.exists(train_path):
        print(f"❌ 训练目录不存在: {train_path}")
        return False
    
    print(f"✅ 训练目录存在: {train_path}")
    
    # 测试训练子目录
    train_subdirs = ['nir', 'cp', 'sk', 'vis']
    for subdir in train_subdirs:
        subdir_path = op.join(train_path, subdir)
        if os.path.exists(subdir_path):
            file_count = len([f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"✅ {subdir}目录存在: {subdir_path} (包含{file_count}个图像文件)")
        else:
            print(f"❌ {subdir}目录不存在: {subdir_path}")
    
    # 测试文本标注文件
    text_anno_path = op.join(train_path, "text_annos.json")
    if os.path.exists(text_anno_path):
        file_size = os.path.getsize(text_anno_path)
        print(f"✅ 文本标注文件存在: {text_anno_path} (大小: {file_size} bytes)")
    else:
        print(f"❌ 文本标注文件不存在: {text_anno_path}")
    
    # 测试验证目录
    val_path = op.join(base_path, "val")
    print(f"\n验证目录: {val_path}")
    
    if not os.path.exists(val_path):
        print(f"❌ 验证目录不存在: {val_path}")
        return False
    
    print(f"✅ 验证目录存在: {val_path}")
    
    # 测试验证子目录
    val_subdirs = ['gallery', 'nir', 'cp', 'sk']
    for subdir in val_subdirs:
        subdir_path = op.join(val_path, subdir)
        if os.path.exists(subdir_path):
            file_count = len([f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"✅ {subdir}目录存在: {subdir_path} (包含{file_count}个图像文件)")
        else:
            print(f"❌ {subdir}目录不存在: {subdir_path}")
    
    # 测试查询文件
    queries_path = op.join(val_path, "val_queries.json")
    if os.path.exists(queries_path):
        file_size = os.path.getsize(queries_path)
        print(f"✅ 查询文件存在: {queries_path} (大小: {file_size} bytes)")
    else:
        print(f"❌ 查询文件不存在: {queries_path}")
    
    print("\n" + "=" * 50)
    print("数据集结构测试完成")
    print("=" * 50)
    
    return True

def test_dataset_import():
    """测试数据集导入"""
    print("\n测试数据集导入...")
    
    try:
        from datasets.prcv import PRCV
        print("✅ PRCV数据集类导入成功")
        
        # 尝试创建数据集实例
        dataset = PRCV(root="/data/taoxuefeng/PRCV")
        print("✅ PRCV数据集实例创建成功")
        
        # 显示数据集信息
        print(f"训练样本数: {len(dataset.train)}")
        print(f"验证查询数: {len(dataset.val['queries'])}")
        print(f"Gallery图像数: {len(dataset.val['img_paths'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始测试PRCV数据集...")
    
    # 测试数据集结构
    structure_ok = test_dataset_structure()
    
    if structure_ok:
        # 测试数据集导入
        import_ok = test_dataset_import()
        
        if import_ok:
            print("\n🎉 所有测试通过！数据集准备就绪。")
            return True
        else:
            print("\n❌ 数据集导入测试失败。")
            return False
    else:
        print("\n❌ 数据集结构测试失败。")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
