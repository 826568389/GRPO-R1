#!/usr/bin/env python3
"""
创建词表大小文件的辅助脚本
用于手动为已训练的模型创建词表大小文件
"""

import os
import argparse
import sys

def create_vocab_size_file(model_path, vocab_size):
    """为模型创建词表大小文件"""
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return False
    
    vocab_size_file = os.path.join(model_path, "vocab_size.txt")
    
    # 检查文件是否已存在
    if os.path.exists(vocab_size_file):
        print(f"警告: 词表大小文件已存在: {vocab_size_file}")
        with open(vocab_size_file, 'r') as f:
            existing_size = f.read().strip()
        print(f"现有词表大小: {existing_size}")
        
        # 询问是否覆盖
        response = input("是否覆盖现有文件? (y/n): ")
        if response.lower() != 'y':
            print("操作已取消")
            return False
    
    # 创建词表大小文件
    try:
        with open(vocab_size_file, 'w') as f:
            f.write(str(vocab_size))
        print(f"成功创建词表大小文件: {vocab_size_file}")
        print(f"词表大小设置为: {vocab_size}")
        return True
    except Exception as e:
        print(f"创建文件时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="为模型创建词表大小文件")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--vocab_size", type=int, required=True, help="词表大小")
    
    args = parser.parse_args()
    
    success = create_vocab_size_file(args.model_path, args.vocab_size)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 