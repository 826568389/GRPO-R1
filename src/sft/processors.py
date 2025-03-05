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
    处理对话格式数据的预处理函数
    
    参数格式示例：
    {
        "instruction": ["指令1", "指令2", ...],
        "input": ["输入1", "输入2", ...],
        "output": ["输出1", "输出2", ...]
    }
    
    返回格式：
    {
        "conversations": ["对话1", "对话2", ...]
    }
    """
    conversations = []
    
    try:
        # 检查数据格式
        if not all(field in examples for field in ["instruction", "input", "output"]):
            raise ValueError("数据格式不正确，需要包含 instruction/input/output 字段")
            
        # 获取数据长度
        batch_size = len(examples["input"])
        
        # 处理每个样本
        for i in range(batch_size):
            try:
                # 确保所有字段都是字符串
                instruction = ensure_string(examples["instruction"][i])
                input_text = ensure_string(examples["input"][i])
                output = ensure_string(examples["output"][i])
                
                # 验证字段非空和长度
                if not all([instruction, input_text, output]):
                    print(f"警告：第{i+1}条数据包含空字段，跳过")
                    continue
                    
                if any(len(text.strip()) < 2 for text in [instruction, input_text, output]):
                    print(f"警告：第{i+1}条数据字段长度过短，跳过")
                    continue
                
                # 构建对话格式
                conversation = (
                    f"<|im_start|>system\n{instruction}<|im_end|>\n"
                    f"<|im_start|>user\n{input_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n{output}<|im_end|>"
                ).strip()
                
                conversations.append(conversation)
                
            except Exception as e:
                print(f"警告：处理第{i+1}条数据时出错：{str(e)}")
                continue
        
        if not conversations:
            print("警告：没有成功处理任何数据")
            
        return {"conversations": conversations}
        
    except Exception as e:
        print(f"错误：预处理函数执行失败 - {str(e)}")
        return {"conversations": []}

def split_dataset(dataset: Dataset, val_ratio: float = 0.1, seed: int = 42) -> DatasetDict:
    """
    将数据集分割为训练集和验证集
    
    参数:
        dataset: 原始数据集
        val_ratio: 验证集比例
        seed: 随机种子
    
    返回:
        包含训练集和验证集的DatasetDict
    """
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 计算数据集大小
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    
    # 随机打乱索引
    indices = np.random.permutation(dataset_size)
    
    # 分割数据集
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # 创建训练集和验证集
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    
    logger.info(f"数据集大小: {dataset_size}")
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

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
