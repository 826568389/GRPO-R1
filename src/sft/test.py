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
from peft import PeftModel, PeftConfig

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
        
        # 检查是否是 LoRA 模型
        adapter_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_path):
            print("检测到 LoRA 模型，正在加载...")
            # 加载 LoRA 配置
            peft_config = PeftConfig.from_pretrained(model_path)
            
            # 加载基础模型
            print("正在加载基础模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                **model_kwargs
            )
            
            # 加载 LoRA 权重
            print("正在加载 LoRA 权重...")
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # 加载完整模型
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
    max_new_tokens: int = 512,  # 限制新生成的token数量
) -> str:
    """生成回复（流式输出）"""
    # 准备输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 存储完整响应
    full_response = ""
    
    # 打印助手开始标记
    print("\n助手: ", end="", flush=True)
    
    # 设置停止条件
    stop_tokens = ["\n\n", "\n\n\n"]  # 多个连续换行通常表示回答结束
    stop_sequences = [tokenizer.encode(s, add_special_tokens=False) for s in stop_tokens]
    
    # 记录重复次数
    repeat_count = 0
    last_tokens = []
    max_repeat_tokens = 5  # 允许重复的最大token数
    
    # 设置初始past_key_values为None
    past = None
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        generated_tokens = 0  # 记录生成的token数量
        
        while generated_tokens < max_new_tokens:
            # 使用past_key_values进行增量生成
            outputs = model(
                input_ids=input_ids[:, -1:] if past is not None else input_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            
            # 更新past_key_values
            past = outputs.past_key_values
            
            # 获取下一个token的概率分布
            next_token_logits = outputs.logits[:, -1, :].float()
            
            # 应用重复惩罚
            if len(last_tokens) > 0:
                for token in set(last_tokens):
                    next_token_logits[0, token] /= repetition_penalty
            
            # 应用temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # 应用top_p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 检查是否生成了结束符
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # 更新last_tokens列表
            last_tokens.append(next_token.item())
            if len(last_tokens) > max_repeat_tokens:
                last_tokens.pop(0)
            
            # 检查重复
            if len(last_tokens) == max_repeat_tokens:
                if len(set(last_tokens)) == 1:  # 所有token都相同
                    repeat_count += 1
                    if repeat_count >= 3:  # 如果连续重复3次，则停止生成
                        break
                else:
                    repeat_count = 0
            
            # 检查停止序列
            for stop_seq in stop_sequences:
                if len(last_tokens) >= len(stop_seq):
                    if last_tokens[-len(stop_seq):] == stop_seq:
                        break
            
            # 解码并打印新生成的token
            text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            if text.strip():
                print(text, end="", flush=True)
                full_response += text
            
            # 更新input_ids和attention_mask
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)
            
            generated_tokens += 1
    
    print("\n")  # 打印换行
    return full_response.strip()

def interactive_test(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 512,
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
            generate_response(
                model, 
                tokenizer, 
                prompt,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
            
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
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度（默认0.7）")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p采样参数（默认0.9）")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大序列长度（默认2048）")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大新生成token数（默认512）")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="重复惩罚系数（默认1.1）")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.ERROR)
    
    model, tokenizer = load_model(
        model_path=args.model_path,
        device=args.device,
        fp16=args.fp16,
    )
    
    interactive_test(
        model, 
        tokenizer, 
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

if __name__ == "__main__":
    main() 