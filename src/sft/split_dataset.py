"""
数据集分割脚本
用于将原始数据集分割成训练集和测试集
"""

import os
import json
import random
import logging
import argparse
from typing import List, Dict

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"解析JSON行时出错: {str(e)}")
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """保存数据为JSONL格式"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def split_dataset(data: List[Dict], test_ratio: float = 0.1, seed: int = 42) -> tuple:
    """分割数据集"""
    random.seed(seed)
    data_size = len(data)
    test_size = int(data_size * test_ratio)
    
    # 随机打乱数据
    indices = list(range(data_size))
    random.shuffle(indices)
    
    # 分割数据
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    test_data = [data[i] for i in test_indices]
    train_data = [data[i] for i in train_indices]
    
    return train_data, test_data

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据集分割工具')
    parser.add_argument('--input_file', type=str, required=True, help='输入数据文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    parser.add_argument('--test_file_name', type=str, required=True, help='测试集文件名')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 加载数据
    logger.info(f"正在加载数据: {args.input_file}")
    data = load_jsonl(args.input_file)
    logger.info(f"加载了 {len(data)} 条数据")
    
    # 分割数据集
    logger.info("正在分割数据集...")
    train_data, test_data = split_dataset(data, test_ratio=args.test_ratio, seed=args.seed)
    
    # 保存分割后的数据集
    test_file = os.path.join(args.output_dir, args.test_file_name)
    save_jsonl(test_data, test_file)
    logger.info(f"测试集已保存到: {test_file}")
    logger.info(f"测试集大小: {len(test_data)} 条数据")
    
    # 打印示例数据
    logger.info("\n测试集示例:")
    for i, item in enumerate(test_data[:2]):
        logger.info(f"\n示例 {i+1}:")
        for key, value in item.items():
            logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main() 