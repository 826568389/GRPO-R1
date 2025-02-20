"""
SFT模型测试脚本
用于加载微调后的模型并进行交互式测试
"""

import os
import logging
import argparse
import warnings
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoConfig,
)

# 禁用警告
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

def load_model(
    model_path: str,
    device: str = "cuda",
    fp16: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """加载模型和分词器"""
    try:
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
        if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
        else:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(
                config,
                **model_kwargs
            )
            
            if os.path.exists(os.path.join(model_path, "model.safetensors")):
                from safetensors.torch import load_file
                state_dict = load_file(os.path.join(model_path, "model.safetensors"))
                
                if "model.embed_tokens.weight" in state_dict:
                    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
                
                model.load_state_dict(state_dict, strict=False)
                
                if hasattr(model, "lm_head") and hasattr(model, "model"):
                    if hasattr(model.model, "embed_tokens"):
                        model.lm_head.weight = model.model.embed_tokens.weight
            else:
                raise ValueError("未找到可加载的模型文件")
        
        # 将模型移动到设备并设置为评估模式
        print(f"正在将模型移动到{device}设备...")
        model = model.to(device).eval()
        print("模型加载完成！\n")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        raise

def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """生成回复（流式输出）"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 存储完整响应
    full_response = ""
    
    # 打印助手开始标记
    print("\n助手: ", end="", flush=True)
    
    # 设置初始past_key_values为None
    past = None
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            # 使用past_key_values进行增量生成
            outputs = model(
                input_ids=input_ids[:, -1:] if past is not None else input_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            
            # 更新past_key_values
            past = outputs.past_key_values
            
            # 获取下一个token的概率分布
            next_token_logits = outputs.logits[:, -1, :]
            
            # 应用temperature和top_p采样
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # 计算top_p过滤的概率分布
            probs = torch.softmax(next_token_logits, dim=-1)
            if top_p > 0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_keep = cumsum_probs <= top_p
                sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
                sorted_indices_to_keep[..., 0] = 1
                indices_to_keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, sorted_indices, sorted_indices_to_keep)
                next_token_logits[~indices_to_keep] = float('-inf')
            
            # 采样下一个token
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            
            # 如果生成了结束符则停止
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # 解码并打印新生成的token
            text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            if text.strip():
                print(text, end="", flush=True)
                full_response += text
            
            # 更新input_ids和attention_mask
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)
    
    print("\n")  # 打印换行
    return full_response.strip()

def interactive_test(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    system_prompt: Optional[str] = None,
):
    """交互式测试"""
    print("\n" + "="*50)
    print("交互式测试模式 (输入 'q' 退出)")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            # 构建完整提示
            if system_prompt:
                prompt = f"{system_prompt}\n\n用户: {user_input}\n助手: "
            else:
                prompt = f"用户: {user_input}\n助手: "
            
            # 生成回复（流式输出）
            generate_response(model, tokenizer, prompt)
            
        except KeyboardInterrupt:
            print("\n已中断生成")
            continue
        except Exception as e:
            logger.error(f"生成回复时出错: {str(e)}")
            continue
    
    print("\n已退出交互式测试")

def main():
    parser = argparse.ArgumentParser(description="SFT模型测试脚本")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="运行设备")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="是否使用 float16 精度（默认开启）")
    parser.add_argument("--system_prompt", type=str,
                        help="系统提示语")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.ERROR)
    
    model, tokenizer = load_model(
        model_path=args.model_path,
        device=args.device,
        fp16=args.fp16,
    )
    
    interactive_test(model, tokenizer, args.system_prompt)

if __name__ == "__main__":
    main() 