#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def test_query_mapping():
    """测试query_idx映射是否正确"""
    
    # 读取val_queries.json
    val_queries_path = '/data/taoxuefeng/PRCV/val/val_queries.json'
    with open(val_queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    print(f"Total queries: {len(queries)}")
    
    # 检查前几个查询
    print("\nFirst 10 queries:")
    for i in range(min(10, len(queries))):
        query = queries[i]
        print(f"Index {i}: query_idx={query['query_idx']}, query_type={query['query_type']}")
    
    # 检查后几个查询
    print("\nLast 10 queries:")
    for i in range(max(0, len(queries)-10), len(queries)):
        query = queries[i]
        print(f"Index {i}: query_idx={query['query_idx']}, query_type={query['query_type']}")
    
    # 查找特定的query_idx
    print("\nLooking for specific query_idx values:")
    target_indices = [5100, 5101, 5102, 5103, 5104]
    for target_idx in target_indices:
        found = False
        for i, query in enumerate(queries):
            if query['query_idx'] == target_idx:
                print(f"query_idx {target_idx} found at index {i}: query_type={query['query_type']}")
                found = True
                break
        if not found:
            print(f"query_idx {target_idx} not found")
    
    # 检查query_type的分布
    print("\nQuery type distribution:")
    type_count = {}
    for query in queries:
        query_type = query['query_type']
        type_count[query_type] = type_count.get(query_type, 0) + 1
    
    for query_type, count in sorted(type_count.items()):
        print(f"{query_type}: {count}")

if __name__ == "__main__":
    test_query_mapping()
