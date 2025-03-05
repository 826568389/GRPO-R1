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
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import torch

# 禁用警告
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_model(
    model_path: str,
    base_model_path: str = None,
    device: str = "cuda",
    fp16: bool = False,
    bf16: bool = False,
    merge_lora: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """加载模型和分词器"""
    print(f"\n正在加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise ValueError(f"模型路径不存在: {model_path}")
    
    # 检查是否为LoRA模型
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora_model = os.path.exists(adapter_config_path)
    
    if is_lora_model:
        if not base_model_path:
            raise ValueError("使用LoRA模型时必须提供基础模型路径")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        
        # 设置模型加载配置
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None,
            "torch_dtype": torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
        }
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs
        )
        
        # 加载LoRA配置和权重
        config = PeftConfig.from_pretrained(model_path)
        
        try:
            model = PeftModel.from_pretrained(
                model,
                model_path,
                torch_dtype=model_kwargs["torch_dtype"],
                is_trainable=False,
            )
        except:
            # 创建新的LoRA配置
            lora_config = LoraConfig(
                r=config.r,
                lora_alpha=config.lora_alpha,
                target_modules=config.target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # 重新创建PeftModel
            model = get_peft_model(model, lora_config)
            
            # 加载权重
            adapter_model_path = os.path.join(model_path, "adapter_model.safetensors")
            if not os.path.exists(adapter_model_path):
                adapter_model_path = os.path.join(model_path, "adapter_model.bin")
            
            if adapter_model_path.endswith(".safetensors"):
                from safetensors import safe_open
                with safe_open(adapter_model_path, framework="pt") as f:
                    weight_map = {key: f.get_tensor(key) for key in f.keys()}
            else:
                weight_map = torch.load(adapter_model_path, map_location="cpu")
            
            model.load_state_dict(weight_map, strict=False)
        
        # 合并LoRA权重
        if merge_lora and hasattr(model, 'merge_and_unload'):
            try:
                model = model.merge_and_unload()
                print("LoRA权重已合并到基础模型")
            except Exception as e:
                print(f"LoRA权重合并失败: {str(e)}")
    else:
        # 加载普通模型
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
        )
    
    # 设置为评估模式
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
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
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("助手回答:", response.strip())
    return response.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型路径，如果是LoRA则是adapter路径")
    parser.add_argument("--base_model_path", type=str, help="基础模型路径，当使用LoRA时必须提供")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", help="使用FP16精度")
    parser.add_argument("--bf16", action="store_true", help="使用BF16精度")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--system_prompt", type=str, default="你是一个专业的助手，请根据用户的问题提供准确、专业的回答。")
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
            bf16=args.bf16,
            merge_lora=args.merge_lora
        )
        
        print("\n开始对话，输入问题后按回车，输入 'quit' 退出")
        print(f"System提示: {args.system_prompt}")
        
        while True:
            try:
                user_input = input("\n用户输入: ")
                if user_input.lower() == 'quit':
                    print("\n对话结束")
                    break
                    
                if user_input.strip():
                    generate_response(
                        model,
                        tokenizer,
                        user_input,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        max_new_tokens=args.max_new_tokens,
                        system_prompt=args.system_prompt,
                    )
            except KeyboardInterrupt:
                print("\n用户中断，如需退出请输入 'quit'")
                continue
        
    except Exception as e:
        print(f"\n运行时错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 