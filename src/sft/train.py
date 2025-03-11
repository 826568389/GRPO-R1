"""
SFT训练入口脚本
实现了数据加载和训练流程
"""

import os
import logging
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

from configs import SFTArguments
from trainer import SFTTrainer
from processors import preprocess_chat_function, normalize_data

logger = logging.getLogger(__name__)


def train():
    try:
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
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
        logger.info("正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
        
        # 准备模型配置
        logger.info("正在准备模型配置...")
        model_config = {
            "use_cache": False,  # 禁用KV缓存以节省显存
            "trust_remote_code": True,
        }
        
        # 根据是否支持bfloat16设置dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            logger.info("使用bfloat16精度")
            model_config["torch_dtype"] = torch.bfloat16
        else:
            logger.info("设备不支持bfloat16，使用float16精度")
            model_config["torch_dtype"] = torch.float16
            
        #设置device_map
        model_config["device_map"] = None if args.deepspeed else "auto"       

        # 加载基础模型
        logger.info("正在加载模型...")
        
            
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_config
        )
        
        # 如果使用DeepSpeed，将模型移动到GPU
        if args.deepspeed:
            model = model.cuda()
            
        # 启用梯度检查点
        if args.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False  # 梯度检查点需要禁用KV缓存
            
        # 加载数据
        data_list = []
        with open(args.dataset_name, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        data_list.append(item)
                    except json.JSONDecodeError:
                        continue
        
        if not data_list:
            raise ValueError("没有成功加载任何数据")
        
        # 数据标准化和转换
        normalized_data = normalize_data(data_list)
        
        # 分割训练集和验证集
        train_data, eval_data = train_test_split(
            normalized_data, 
            test_size=0.2, 
            random_state=args.seed
        )
        
        # 创建数据集，使用streaming模式
        train_dataset = Dataset.from_list(train_data).with_format("torch")
        eval_dataset = Dataset.from_list(eval_data).with_format("torch")
        
        # 处理对话格式
        train_dataset = train_dataset.map(
            preprocess_chat_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=False,  # 禁用缓存
        )
        
        eval_dataset = eval_dataset.map(
            preprocess_chat_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False,  # 禁用缓存
        )
        
        # 对数据集进行tokenize处理
        def tokenize_function(examples):
            if "conversations" not in examples:
                raise ValueError("数据集中缺少'conversations'字段")
            
            tokenized = tokenizer(
                examples["conversations"],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            labels = tokenized["input_ids"].clone()
            
            for i in range(len(examples["conversations"])):
                conversation = examples["conversations"][i]
                assistant_marker = "<|assistant|>"
                if assistant_marker in conversation:
                    assistant_pos = conversation.find(assistant_marker)
                    user_text = conversation[:assistant_pos]
                    user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
                    if len(user_tokens) > 0:
                        labels[i, :len(user_tokens)] = -100
                else:
                    seq_len = (tokenized["attention_mask"][i] == 1).sum()
                    if isinstance(seq_len, torch.Tensor):
                        seq_len = seq_len.item()
                    labels[i, :int(seq_len * 0.6)] = -100
            
            # 设置padding部分的标签为-100
            attention_mask = tokenized["attention_mask"]
            labels[attention_mask == 0] = -100
            
            tokenized["labels"] = labels
            return tokenized
        
        # 应用tokenize处理，使用较小的batch_size
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,  # 使用较小的batch_size处理数据
            remove_columns=train_dataset.column_names,
            load_from_cache_file=False,
        )
        
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,  # 使用较小的batch_size处理数据
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False,
        )
        
        # 设置数据集格式
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # 如果使用LoRA，添加配置
        if args.training_mode == "lora":
            for param in model.parameters():
                param.requires_grad = False
            
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules.split(","),
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
            )
            model = get_peft_model(model, lora_config)
        
        # 创建训练器并开始训练
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=None,
        )
        
        trainer.train()
        
        # 保存模型
        save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_path, exist_ok=True)
        
        if args.training_mode == "lora":
            trainer.accelerator.wait_for_everyone()
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
            if trainer.accelerator.is_main_process:
                try:
                    unwrapped_model.save_pretrained(save_path, save_function=trainer.accelerator.save)
                    tokenizer.save_pretrained(save_path)
                except Exception as e:
                    logger.error(f"保存模型时出错: {str(e)}")
        else:
            trainer.save_model(save_path)
            
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}")
        logger.debug("错误详情:", exc_info=True)
    finally:
        logger.info("训练脚本执行完成")

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    train() 