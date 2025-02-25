"""
SFT模型测试脚本
用于加载微调后的模型并进行一问一答测试
"""

import os
import json
import logging
import argparse
import warnings
from typing import Tuple

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 禁用警告
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_model(
    model_path: str,
    base_model_path: str = None,
    device: str = "cuda",
    fp16: bool = True,
    merge_lora: bool = False,  # 新增参数，控制是否合并LoRA权重
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """加载模型和分词器"""
    print(f"\n正在加载模型: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise ValueError(f"模型路径不存在: {model_path}")
    
    # 检查是否为LoRA模型
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora_model = os.path.exists(adapter_config_path)
    
    if is_lora_model:
        if not base_model_path:
            raise ValueError("使用LoRA模型时必须提供基础模型路径")
            
        print(f"检测到LoRA模型")
        print(f"基础模型路径: {base_model_path}")
        print(f"LoRA权重路径: {model_path}")
        
        # 检查权重文件
        adapter_model_path = os.path.join(model_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_model_path):
            adapter_model_path = os.path.join(model_path, "adapter_model.bin")
        if not os.path.exists(adapter_model_path):
            raise ValueError(f"未找到LoRA权重文件！检查路径: {model_path}")
        else:
            print(f"找到LoRA权重文件: {os.path.basename(adapter_model_path)}")
        
        # 加载分词器
        print("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        
        # 设置模型加载配置
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if fp16 else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
        }
        
        # 加载基础模型
        print("正在加载基础模型...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs
        )
        
        # 加载LoRA权重
        print("正在加载LoRA权重...")
        from peft import PeftConfig
        config = PeftConfig.from_pretrained(model_path)
        print("\nLoRA模型信息:")
        print(f"基础模型: {config.base_model_name_or_path}")
        print(f"任务类型: {config.task_type}")
        print(f"LoRA秩 (r): {config.r}")
        print(f"LoRA alpha: {config.lora_alpha}")
        print(f"目标模块: {config.target_modules}")
        
        # 确保模型配置与训练时一致
        if not config.target_modules or len(config.target_modules) == 0:
            print("警告: LoRA配置中未找到目标模块，使用默认配置")
            config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj"]
            config.r = 8
            config.lora_alpha = 16
            
        # 加载LoRA模型
        model = PeftModel.from_pretrained(
            model,
            model_path,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            is_trainable=False,  # 设置为推理模式
        )
        
        # 确保模型处于正确的模式
        model.eval()
        
        # 根据参数决定是否合并LoRA权重
        if merge_lora and hasattr(model, 'merge_and_unload'):
            print("合并LoRA权重到基础模型...")
            try:
                model = model.merge_and_unload()
                print("LoRA权重合并成功")
            except Exception as e:
                print(f"警告: LoRA权重合并失败 - {str(e)}")
                print("继续使用未合并的模型")
        else:
            print("使用动态LoRA权重（未合并）")
            
    else:
        # 加载分词器
        print("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        # 设置模型加载配置
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if fp16 else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
        }
        
        # 加载模型
        print("正在加载模型权重...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
    
    # 将模型移动到设备并设置为评估模式
    if device == "cuda" and not model_kwargs.get("device_map"):
        print(f"正在将模型移动到{device}设备...")
        model = model.to(device)
    model = model.eval()
    print("模型加载完成！\n")
    return model, tokenizer

def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    temperature: float = 0.1,
    top_p: float = 0.7,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 256,
    system_prompt: str = "作为网络安全领域的专家，你专注于解决各类网络安全问题，能全面分析解决网络攻击、漏洞管理、数据保护、身份认证、云安全、事件响应、零信任架构、物联网安全、APT防护及法律合规等网络安全相关问题，并提供详细的处置或技术建议",
) -> str:
    """生成回复"""
    # 构建system和user输入
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    # 使用chat模板构建输入
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 准备模型输入
    model_inputs = tokenizer([prompt], return_tensors="pt")
    
    # 确保输入在正确的设备上
    for k, v in model_inputs.items():
        model_inputs[k] = v.to(model.device)
    
    with torch.no_grad():
        # 确保模型在评估模式
        model.eval()
        
        # 生成回复
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
        )
        
        # 只保留新生成的token
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("助手回答:", response.strip())
    return response.strip()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型路径，如果是LoRA则是adapter路径")
    parser.add_argument("--base_model_path", type=str, help="基础模型路径，当使用LoRA时必须提供")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--system_prompt", type=str, default="你是一个专业的助手，请根据用户的问题提供准确、专业的回答。", 
                      help="System prompt for the model")
    args = parser.parse_args()

    try:
        # 检查是否为LoRA模型
        adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
        is_lora_model = os.path.exists(adapter_config_path)
        
        if is_lora_model and not args.base_model_path:
            raise ValueError("使用LoRA模型时必须提供基础模型路径 (--base_model_path)")
        
        # 加载模型和分词器
        model, tokenizer = load_model(
            model_path=args.model_path,
            base_model_path=args.base_model_path if is_lora_model else None,
            device=args.device,
            fp16=args.fp16,
            merge_lora=args.merge_lora
        )
        
        print("\n开始对话，输入问题后按回车，输入 'quit' 退出")
        print(f"System提示: {args.system_prompt}")
        
        while True:
            try:
                # 等待用户输入
                user_input = input("\n用户输入: ")
                
                # 检查是否退出
                if user_input.lower() == 'quit':
                    print("\n对话结束")
                    break
                    
                # 只有当用户实际输入内容时才生成回答
                if user_input.strip():
                    generate_response(
                        model,
                        tokenizer,
                        user_input,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        max_new_tokens=args.max_new_tokens,
                        system_prompt=args.system_prompt,  # 传递system提示
                    )
            except KeyboardInterrupt:
                # Ctrl+C 处理
                print("\n用户中断，如需退出请输入 'quit'")
                continue
        
    except Exception as e:
        print(f"\n运行时错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 