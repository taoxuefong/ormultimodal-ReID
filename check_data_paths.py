#!/usr/bin/env python3
"""
检查PRCV数据集路径的脚本
"""

import os
import os.path as op
import json

def check_train_data():
    """检查训练数据路径"""
    print("=" * 50)
    print("检查训练数据路径")
    print("=" * 50)
    
    base_path = "/data/taoxuefeng/PRCV"
    train_path = op.join(base_path, "train")
    text_anno_path = op.join(train_path, "text_annos.json")
    
    print(f"基础路径: {base_path}")
    print(f"训练路径: {train_path}")
    print(f"文本标注路径: {text_anno_path}")
    
    # 检查文本标注文件
    if not os.path.exists(text_anno_path):
        print(f"❌ 文本标注文件不存在: {text_anno_path}")
        return False
    
    print(f"✅ 文本标注文件存在")
    
    # 读取文本标注
    with open(text_anno_path, 'r', encoding='utf-8') as f:
        annos = json.load(f)
    
    print(f"✅ 读取到 {len(annos)} 个标注")
    
    # 检查前几个样本的路径
    print("\n检查前5个样本的路径:")
    for i, anno in enumerate(annos[:5]):
        pid = anno['id']
        file_path = anno['file_path']
        caption = anno['caption'][:50] + "..." if len(anno['caption']) > 50 else anno['caption']
        
        # 构建完整路径
        full_path = op.join(train_path, file_path)
        
        print(f"\n样本 {i+1}:")
        print(f"  ID: {pid}")
        print(f"  文件路径: {file_path}")
        print(f"  完整路径: {full_path}")
        print(f"  路径存在: {'✅' if os.path.exists(full_path) else '❌'}")
        print(f"  描述: {caption}")
        
        if not os.path.exists(full_path):
            print(f"  ❌ 文件不存在，检查父目录...")
            parent_dir = op.dirname(full_path)
            if os.path.exists(parent_dir):
                print(f"  ✅ 父目录存在: {parent_dir}")
                files = os.listdir(parent_dir)
                print(f"  父目录中的文件: {files[:5]}...")
            else:
                print(f"  ❌ 父目录不存在: {parent_dir}")
    
    return True

def check_val_data():
    """检查验证数据路径"""
    print("\n" + "=" * 50)
    print("检查验证数据路径")
    print("=" * 50)
    
    base_path = "/data/taoxuefeng/PRCV"
    val_path = op.join(base_path, "val")
    queries_path = op.join(val_path, "val_queries.json")
    
    print(f"验证路径: {val_path}")
    print(f"查询文件路径: {queries_path}")
    
    # 检查查询文件
    if not os.path.exists(queries_path):
        print(f"❌ 查询文件不存在: {queries_path}")
        return False
    
    print(f"✅ 查询文件存在")
    
    # 读取查询文件
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    print(f"✅ 读取到 {len(queries)} 个查询")
    
    # 检查前几个查询
    print("\n检查前3个查询:")
    for i, query in enumerate(queries[:3]):
        query_idx = query['query_idx']
        query_type = query['query_type']
        content = query['content']
        
        print(f"\n查询 {i+1}:")
        print(f"  索引: {query_idx}")
        print(f"  类型: {query_type}")
        print(f"  内容: {content}")
        
        # 检查内容中的文件路径
        for j, item in enumerate(content):
            if isinstance(item, str) and ('nir/' in item or 'cp/' in item or 'sk/' in item):
                full_path = op.join(val_path, item)
                print(f"    文件 {j+1}: {item}")
                print(f"    完整路径: {full_path}")
                print(f"    路径存在: {'✅' if os.path.exists(full_path) else '❌'}")
    
    return True

def check_directory_structure():
    """检查目录结构"""
    print("\n" + "=" * 50)
    print("检查目录结构")
    print("=" * 50)
    
    base_path = "/data/taoxuefeng/PRCV"
    
    # 检查训练目录结构
    train_path = op.join(base_path, "train")
    if os.path.exists(train_path):
        print(f"✅ 训练目录存在: {train_path}")
        train_subdirs = os.listdir(train_path)
        print(f"  训练子目录: {train_subdirs}")
        
        # 检查vis目录
        vis_path = op.join(train_path, "vis")
        if os.path.exists(vis_path):
            print(f"✅ vis目录存在: {vis_path}")
            vis_subdirs = os.listdir(vis_path)
            print(f"  vis子目录数量: {len(vis_subdirs)}")
            print(f"  前10个vis子目录: {vis_subdirs[:10]}")
            
            # 检查一个具体的子目录
            if vis_subdirs:
                sample_subdir = vis_subdirs[0]
                sample_path = op.join(vis_path, sample_subdir)
                if os.path.isdir(sample_path):
                    files = os.listdir(sample_path)
                    print(f"  示例子目录 {sample_subdir} 包含 {len(files)} 个文件")
                    if files:
                        print(f"  示例文件: {files[0]}")
        else:
            print(f"❌ vis目录不存在: {vis_path}")
    else:
        print(f"❌ 训练目录不存在: {train_path}")
    
    # 检查验证目录结构
    val_path = op.join(base_path, "val")
    if os.path.exists(val_path):
        print(f"✅ 验证目录存在: {val_path}")
        val_subdirs = os.listdir(val_path)
        print(f"  验证子目录: {val_subdirs}")
    else:
        print(f"❌ 验证目录不存在: {val_path}")

def main():
    """主函数"""
    print("开始检查PRCV数据集路径...")
    
    try:
        check_train_data()
        check_val_data()
        check_directory_structure()
        
        print("\n" + "=" * 50)
        print("路径检查完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 检查过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
