"""
数据预处理模块
包含数据集的预处理函数
"""

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
    """
    normalized_data = []
    for idx, item in enumerate(data_list, 1):
        try:
            # 检查必要字段
            if not all(key in item for key in ["instruction", "input", "output"]):
                print(f"警告：第{idx}条数据缺少必要字段，跳过")
                continue
                
            # 标准化字段
            normalized_item = {
                "instruction": ensure_string(item["instruction"]),
                "input": ensure_string(item["input"]),
                "output": ensure_string(item["output"])
            }
            
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
    return normalized_data

def preprocess_chat_function(examples):
    """
    处理对话格式数据的预处理函数
    """
    conversations = []
    
    try:
        # 检查数据格式
        if all(field in examples for field in ["input", "output", "instruction"]):
            # 处理CVE_QA格式
            for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
                try:
                    # 确保所有字段都是字符串
                    instruction_text = ensure_string(instruction)
                    input_text = ensure_string(input_text)
                    output_text = ensure_string(output)
                    
                    # 构建对话格式
                    conversation = f"<|im_start|>system\n{instruction_text}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>\n"
                    conversations.append(conversation.strip())
                    
                except Exception as e:
                    print(f"警告：处理单条数据时出错：{str(e)}")
                    continue
                    
        else:
            raise ValueError("数据格式不正确，需要包含 input/output/instruction 字段")
        
        return {"conversations": conversations}
        
    except Exception as e:
        print(f"错误：预处理函数执行失败 - {str(e)}")
        return {"conversations": []} 