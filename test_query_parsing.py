#!/usr/bin/env python3
"""
测试query_type解析和content映射的逻辑
"""

def test_query_parsing():
    """测试query_type解析逻辑"""
    
    def parse_query_type(query_type: str):
        """模拟PRCV数据集中的_parse_query_type方法"""
        modalities = []
        # 按照query_type字符串中出现的顺序来添加模态
        if 'TEXT' in query_type:
            modalities.append('text')
        if 'NIR' in query_type:
            modalities.append('nir')
        if 'CP' in query_type:
            modalities.append('cp')
        if 'SK' in query_type:
            modalities.append('sk')
        return modalities
    
    # 测试用例
    test_cases = [
        "onemodal_NIR",
        "onemodal_CP", 
        "onemodal_SK",
        "onemodal_TEXT",
        "twomodal_NIR_CP",
        "twomodal_CP_SK",
        "twomodal_TEXT_NIR",
        "twomodal_TEXT_SK",
        "threemodal_CP_NIR_TEXT",
        "fourmodal_TEXT_NIR_CP_SK"
    ]
    
    print("测试query_type解析:")
    for query_type in test_cases:
        modalities = parse_query_type(query_type)
        print(f"  {query_type} -> {modalities}")
    
    print("\n测试content映射逻辑:")
    
    # 模拟一个twomodal_TEXT_SK的查询
    query_type = "twomodal_TEXT_SK"
    modalities = parse_query_type(query_type)
    content = ["A young man of medium build...", "sk/515.jpg"]
    
    print(f"Query type: {query_type}")
    print(f"Modalities: {modalities}")
    print(f"Content: {content}")
    
    # 创建内容映射
    content_map = {}
    for i, modality in enumerate(modalities):
        content_map[modality] = content[i]
    
    print(f"Content map: {content_map}")
    
    # 验证映射是否正确
    print("\n验证映射:")
    for modality in modalities:
        mapped_content = content_map[modality]
        print(f"  {modality} -> {mapped_content}")
        
        # 检查内容类型
        if modality == 'text':
            print(f"    (文本内容，长度: {len(mapped_content)})")
        elif modality in ['nir', 'cp', 'sk']:
            print(f"    (图像路径: {mapped_content})")

if __name__ == '__main__':
    test_query_parsing()
