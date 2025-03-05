"""
SFT模型评估脚本
用于评估训练好的模型性能，包括困惑度、准确率等指标，以及生成测试
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
import wandb

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(args):
    """加载模型和分词器"""
    logger.info(f"正在加载模型: {args.model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path if args.base_model_path else args.model_path,
        trust_remote_code=True,
    )
    
    # 设置模型加载配置
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if args.device == "cuda" else None,
        "torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    }
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path if args.base_model_path else args.model_path,
        **model_kwargs
    )
    
    # 如果是LoRA模型，加载adapter
    if args.base_model_path:
        model = PeftModel.from_pretrained(
            model,
            args.model_path,
            torch_dtype=model_kwargs["torch_dtype"]
        )
    
    model.eval()
    logger.info("模型加载完成")
    return model, tokenizer

def load_test_data(args):
    """加载测试数据"""
    logger.info(f"正在加载测试数据: {args.test_file}")
    
    if args.test_file.endswith('.jsonl'):
        # 加载JSONL格式数据
        with open(args.test_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        # 加载JSON格式数据
        with open(args.test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # 确保数据格式正确
    processed_data = []
    for item in data:
        if isinstance(item, dict) and 'instruction' in item and 'input' in item and 'output' in item:
            processed_data.append(item)
    
    dataset = Dataset.from_list(processed_data)
    logger.info(f"加载了 {len(dataset)} 条测试数据")
    return dataset

def preprocess_function(examples, tokenizer, max_length):
    """预处理数据"""
    conversations = []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        conversation = (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        conversations.append(conversation)
    
    # tokenize
    tokenized = tokenizer(
        conversations,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    tokenized["labels"] = tokenized["input_ids"].clone()
    tokenized["labels"][tokenized["input_ids"] == tokenizer.pad_token_id] = -100
    
    return tokenized

def compute_metrics(predictions, labels):
    """计算评估指标"""
    # 确保输入是一维数组
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    # 计算F1分数
    f1 = f1_score(labels, predictions, average='weighted')
    
    # 计算分类报告
    report = classification_report(
        labels, 
        predictions,
        output_dict=True
    )
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "classification_report": report
    }

def generate_response(model, tokenizer, instruction: str, input_text: str, args) -> str:
    """生成回复"""
    conversation = (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def evaluate_model(model, tokenizer, dataset, args):
    """评估模型性能"""
    logger.info("开始评估模型...")
    
    # 初始化指标
    total_loss = 0
    all_predictions = []
    all_labels = []
    generated_responses = []
    
    # 评估循环
    for idx in tqdm(range(len(dataset)), desc="评估进度"):
        example = dataset[idx]
        
        # 计算困惑度
        inputs = preprocess_function(
            {"instruction": [example["instruction"]], 
             "input": [example["input"]], 
             "output": [example["output"]]},
            tokenizer,
            args.max_length
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            
            if args.generate:
                # 生成回复
                response = generate_response(
                    model, 
                    tokenizer,
                    example["instruction"],
                    example["input"],
                    args
                )
                generated_responses.append({
                    "input": example["input"],
                    "target": example["output"],
                    "prediction": response
                })
            
            # 收集预测结果（每个样本单独处理）
            logits = outputs.logits[0]  # 只取第一个样本，因为batch_size=1
            predictions = torch.argmax(logits, dim=-1)
            labels = inputs["labels"][0]  # 同样只取第一个样本
            
            # 只保留非填充部分的预测和标签
            mask = labels != -100
            filtered_preds = predictions[mask].cpu().numpy()
            filtered_labels = labels[mask].cpu().numpy()
            
            all_predictions.extend(filtered_preds)
            all_labels.extend(filtered_labels)
    
    # 计算平均损失和困惑度
    avg_loss = total_loss / len(dataset)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # 确保预测结果和标签长度匹配
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    if len(all_predictions) != len(all_labels):
        logger.warning(f"预测结果和标签长度不匹配: predictions={len(all_predictions)}, labels={len(all_labels)}")
        # 取最小长度
        min_len = min(len(all_predictions), len(all_labels))
        all_predictions = all_predictions[:min_len]
        all_labels = all_labels[:min_len]
    
    # 计算其他指标
    metrics = compute_metrics(all_predictions, all_labels)
    metrics.update({
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": len(all_predictions)
    })
    
    # 打印评估结果
    logger.info("\n评估结果:")
    logger.info(f"损失: {avg_loss:.4f}")
    logger.info(f"困惑度: {perplexity:.4f}")
    logger.info(f"准确率: {metrics['accuracy']:.4f}")
    logger.info(f"F1分数: {metrics['f1']:.4f}")
    logger.info(f"总token数: {len(all_predictions)}")
    
    # 保存生成结果
    if args.generate:
        output_file = os.path.join(args.output_dir, "generation_results.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in generated_responses:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"生成结果已保存到: {output_file}")
    
    # 保存评估指标
    metrics_file = os.path.join(args.output_dir, "eval_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"评估指标已保存到: {metrics_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--base_model_path", type=str, help="基础模型路径（如果是LoRA）")
    parser.add_argument("--test_file", type=str, required=True, help="测试数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备类型")
    parser.add_argument("--fp16", action="store_true", help="是否使用FP16")
    parser.add_argument("--bf16", action="store_true", help="是否使用BF16")
    
    # 生成参数
    parser.add_argument("--generate", action="store_true", help="是否进行生成测试")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样")
    parser.add_argument("--num_beams", type=int, default=4, help="束搜索大小")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    
    # wandb配置
    parser.add_argument("--wandb_project", type=str, default="model-evaluation")
    parser.add_argument("--wandb_name", type=str, help="wandb运行名称")
    parser.add_argument("--no_wandb", action="store_true", help="禁用wandb")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "eval.log"))
        ]
    )
    
    # 初始化wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"eval-{os.path.basename(args.model_path)}",
            config=vars(args)
        )
    
    try:
        # 加载模型和tokenizer
        model, tokenizer = load_model_and_tokenizer(args)
        
        # 加载测试数据
        test_dataset = load_test_data(args)
        
        # 评估模型
        metrics = evaluate_model(model, tokenizer, test_dataset, args)
        
        # 记录到wandb
        if not args.no_wandb:
            wandb.log(metrics)
        
    except Exception as e:
        logger.error(f"评估过程出错: {str(e)}")
        raise
    finally:
        if not args.no_wandb:
            wandb.finish()

if __name__ == "__main__":
    main() 