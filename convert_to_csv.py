#!/usr/bin/env python3
"""
将val_ranking_results.txt转换为CSV格式
"""

import csv
import os

def convert_txt_to_csv(txt_file_path, csv_file_path):
    """将制表符分隔的txt文件转换为CSV格式"""
    
    # 检查输入文件是否存在
    if not os.path.exists(txt_file_path):
        print(f"错误：找不到文件 {txt_file_path}")
        return
    
    # 创建CSV文件
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        # 创建CSV writer
        fieldnames = ['query_idx', 'query_type', 'ranking_list_idx']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 读取txt文件并转换
        with open(txt_file_path, 'r', encoding='utf-8') as txtfile:
            for line_num, line in enumerate(txtfile, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 按制表符分割行
                parts = line.split('\t')
                if len(parts) != 3:
                    print(f"警告：第{line_num}行格式不正确，期望3列但得到{len(parts)}列，跳过: {line}")
                    continue
                
                try:
                    query_idx = int(parts[0]) + 1  # 原始query_idx递增1
                    query_type = parts[1].strip()
                    ranking_list_idx = parts[2].strip()
                    
                    # 写入CSV行
                    writer.writerow({
                        'query_idx': query_idx,
                        'query_type': query_type,
                        'ranking_list_idx': ranking_list_idx
                    })
                    
                except Exception as e:
                    print(f"警告：处理第{line_num}行时出错: {e}")
                    print(f"行内容: {line}")
                    continue
        
        print(f"转换完成！")
        print(f"CSV文件已保存到: {csv_file_path}")

if __name__ == "__main__":
    # 输入和输出文件路径
    txt_file = "./outputs/val_ranking_results.txt"
    csv_file = "./outputs/val_ranking_results.csv"
    
    print(f"开始转换 {txt_file} 到 {csv_file}...")
    convert_txt_to_csv(txt_file, csv_file)
