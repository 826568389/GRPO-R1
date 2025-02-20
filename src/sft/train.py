"""
SFT训练入口脚本
实现了数据加载和训练流程
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    BitsAndBytesConfig,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

from configs import SFTArguments
from trainer import SFTTrainer

logger = logging.getLogger(__name__)


def train():
    # 解析命令行参数
    parser = HfArgumentParser(SFTArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置运行名称
    if not args.run_name:
        model_name = args.model_name_or_path.split('/')[-1]
        args.run_name = f"sft-{model_name}-{args.training_mode}-{args.learning_rate}"
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    
    # 准备模型配置
    model_config = {
        "use_cache": False if args.use_gradient_checkpointing else True,
        "trust_remote_code": True,
    }
    
    # 设置数据类型
    if args.bf16:
        model_config["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_config["torch_dtype"] = torch.float16
    
    # 设置量化配置（用于QLoRA）
    if args.training_mode == "qlora" and args.quantization_bits:
        model_config["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.quantization_bits == 4,
            load_in_8bit=args.quantization_bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_config
    )
    
    # 配置LoRA或QLoRA
    if args.training_mode in ["lora", "qlora"]:
        if args.training_mode == "qlora":
            model = prepare_model_for_kbit_training(
                model, 
                use_gradient_checkpointing=args.use_gradient_checkpointing
            )
        
        lora_config = LoraConfig(
            r=args.lora_config.r,
            lora_alpha=args.lora_config.alpha,
            target_modules=args.lora_config.target_modules,
            lora_dropout=args.lora_config.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数信息
        model.print_trainable_parameters()
    
    # 加载数据集
    dataset = load_dataset(args.dataset_name)
    
    def preprocess_function(examples):
        """数据预处理函数"""
        # 构建输入文本
        conversations = []
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            if input_text.strip():
                full_text = f"指令：{instruction}\n输入：{input_text}\n输出：{output}"
            else:
                full_text = f"指令：{instruction}\n输出：{output}"
            conversations.append(full_text)
            
        # 使用tokenizer处理文本
        inputs = tokenizer(
            conversations,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # 创建标签
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs
    
    # 预处理数据集
    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    if args.training_mode == "full":
        # 全量模型保存
        trainer.save_model(args.output_dir)
        logger.info(f"模型已保存到: {args.output_dir}")
    elif args.training_mode in ["lora", "qlora"]:
        # LoRA模型保存
        trainer.model.save_pretrained(args.output_dir)
        # 保存分词器
        if trainer.tokenizer is not None:
            trainer.tokenizer.save_pretrained(args.output_dir)
        logger.info(f"LoRA权重已保存到: {args.output_dir}")
    

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    train() 