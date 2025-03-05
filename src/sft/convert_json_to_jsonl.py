#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将JSON数组格式转换为JSONL格式（每行一个JSON对象）
"""

import json
import argparse
import os

def convert_json_to_jsonl(input_file, output_file):
    """
    将JSON数组转换为JSONL格式
    
    Args:
        input_file: 输入文件路径（JSON数组）
        output_file: 输出文件路径（JSONL格式）
    """
    print(f"正在读取JSON文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"输入文件不是JSON数组格式，实际类型: {type(data)}")
    
    print(f"读取到 {len(data)} 条数据")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"正在写入JSONL文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成，已写入 {len(data)} 条数据到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description="将JSON数组转换为JSONL格式")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径（JSON数组）")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径（JSONL格式）")
    
    args = parser.parse_args()
    convert_json_to_jsonl(args.input, args.output)

if __name__ == "__main__":
    main() 