"""
SFT模型测试脚本
用于加载微调后的模型并进行一问一答测试
"""

import os
import logging
import argparse
import warnings
from typing import Tuple

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

# 禁用警告
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_model(
    model_path: str,
    device: str = "cuda",
    fp16: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """加载模型和分词器"""
    print(f"\n正在加载模型: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise ValueError(f"模型路径不存在: {model_path}")
        
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
    }
    
    # 加载模型
    print("正在加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    # 将模型移动到设备并设置为评估模式
    print(f"正在将模型移动到{device}设备...")
    model = model.to(device).eval()
    print("模型加载完成！\n")
    return model, tokenizer

def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 1024,
) -> str:
    """生成回复"""
    # 构建对话消息
    messages = [
        {"role": "system", "content": "你是一个智能助手，请根据用户输入的问题进行回答。"},
        {"role": "user", "content": user_input}
    ]
    
    # 应用对话模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 准备模型输入
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
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
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    args = parser.parse_args()

    try:
        # 加载模型和分词器
        model, tokenizer = load_model(
            model_path=args.model_path,
            device=args.device,
            fp16=args.fp16
        )
        
        print("\n开始对话，输入问题后按回车，输入 'quit' 退出")
        
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