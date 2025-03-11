"""
数据预处理模块
包含数据集的预处理函数
"""

import os
import json
import logging
import torch
from datasets import Dataset, DatasetDict
import numpy as np

logger = logging.getLogger(__name__)

def ensure_string(value):
    """确保值是字符串类型，并处理多行文本"""
    try:
        if value is None:
            return ""
        elif isinstance(value, (str, int, float, bool)):
            return str(value).strip()
        elif isinstance(value, dict):
            # 如果是字典，尝试获取可能的文本字段
            for key in ['text', 'content', 'value', 'answer']:
                if key in value:
                    return str(value[key]).strip()
            # 如果没有找到文本字段，转换整个字典
            return str(value).strip()
        elif isinstance(value, list):
            # 如果是列表，将所有元素连接
            return ' '.join(ensure_string(item) for item in value).strip()
        else:
            # 其他类型直接转换为字符串
            return str(value).strip()
    except Exception as e:
        print(f"警告：字符串转换失败 - {str(e)}")
        return ""

def normalize_data(data_list):
    """
    标准化数据格式，确保所有字段都是字符串类型
    支持两种数据格式：
    1. instruction/input/output 格式
    2. messages 格式
    """
    normalized_data = []
    for idx, item in enumerate(data_list, 1):
        try:
            # 检查数据格式
            if "messages" in item:
                # messages格式
                messages = item.get("messages", [])
                if not isinstance(messages, list) or len(messages) < 2:
                    print(f"警告：第{idx}条数据的messages格式不正确或不完整，跳过")
                    continue
                
                # 提取用户输入和助手回复
                user_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
                assistant_msg = next((msg for msg in messages if msg.get("role") == "assistant"), None)
                
                if not user_msg or not assistant_msg:
                    print(f"警告：第{idx}条数据缺少用户输入或助手回复，跳过")
                    continue
                
                # 构建标准格式
                normalized_item = {
                    "instruction": "回答以下问题，先给出思考过程，再给出最终答案。",
                    "input": ensure_string(user_msg.get("content", "")),
                    "output": ensure_string(assistant_msg.get("content", ""))
                }
                
            elif all(key in item for key in ["instruction", "input", "output"]):
                # 标准格式
                normalized_item = {
                    "instruction": ensure_string(item["instruction"]),
                    "input": ensure_string(item["input"]),
                    "output": ensure_string(item["output"])
                }
            else:
                print(f"警告：第{idx}条数据格式不正确，需要包含messages或instruction/input/output字段，跳过")
                continue
                
            # 验证字段非空
            if not all(normalized_item.values()):
                print(f"警告：第{idx}条数据包含空字段，跳过")
                continue
                
            # 验证字段长度
            if any(len(v.strip()) < 2 for v in normalized_item.values()):
                print(f"警告：第{idx}条数据字段长度过短，跳过")
                continue
                
            normalized_data.append(normalized_item)
            
        except Exception as e:
            print(f"警告：第{idx}条数据标准化失败 - {str(e)}")
            continue
            
    print(f"数据标准化完成：输入 {len(data_list)} 条，输出 {len(normalized_data)} 条有效数据")
    
    # 保存标准化后的数据到文件
    try:
        import os
        import json
        from datetime import datetime
        
        # 创建输出目录
        output_dir = "normalized_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"normalized_data_{timestamp}.jsonl")
        
        # 写入数据
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in normalized_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        print(f"标准化数据已保存到: {output_file}")
        print(f"数据条数: {len(normalized_data)}")
        
    except Exception as e:
        print(f"警告：保存标准化数据时出错 - {str(e)}")
    
    return normalized_data

def preprocess_chat_function(examples):
    """
    处理对话数据，将其转换为模型可接受的格式
    """
    # 初始化结果列表
    conversations_list = []
    
    # 检查必要的字段
    required_fields = ["input", "output"]
    for field in required_fields:
        if field not in examples:
            raise ValueError(f"数据集缺少必要的字段: {field}")
    
    # 处理每个样本
    for i in range(len(examples["input"])):
        # 获取输入和输出
        user_input = examples["input"][i]
        assistant_output = examples["output"][i]
        
        # 确保输入和输出都是字符串
        if not isinstance(user_input, str):
            user_input = str(user_input)
        if not isinstance(assistant_output, str):
            assistant_output = str(assistant_output)
        
        # 构建对话文本，确保格式正确
        conversation = f"<|user|>\n{user_input.strip()}\n<|assistant|>\n{assistant_output.strip()}"
        conversations_list.append(conversation)
        
        # 打印前几个样本，帮助调试
        if i < 2:
            print(f"样本 {i+1}:")
            print(f"用户输入: {user_input[:50]}...")
            print(f"助手输出: {assistant_output[:50]}...")
            print(f"构建的对话: {conversation[:100]}...")
            print("-" * 50)
    
    # 返回处理后的数据
    return {"conversations": conversations_list}

def load_and_preprocess_data(args):
    """加载和预处理数据，返回原始数据集"""
    if args.dataset_name.endswith('.jsonl'):
        # 直接读取整个JSONL文件
        data_list = []
        with open(args.dataset_name, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        data_list.append(item)
                    except json.JSONDecodeError:
                        continue
        
        # 使用normalize_data处理数据
        normalized_data = normalize_data(data_list)
        
        # 创建数据集
        dataset = Dataset.from_list(normalized_data)
    else:
        dataset = load_dataset(args.dataset_name)
    
    # 分割数据集
    datasets = split_dataset(
        dataset=dataset,
        val_ratio=0.1,
        seed=args.seed
    )
    
    return datasets
