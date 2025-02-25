"""
SFT训练入口脚本
实现了数据加载和训练流程
"""

import os
import sys
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    AutoConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

from configs import SFTArguments
from trainer import SFTTrainer
from processors import preprocess_chat_function, normalize_data
from datasets import load_dataset
import torch

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
            "_attn_implementation": "eager",
        }
        
        # 设置数据类型
        model_config["torch_dtype"] = torch.bfloat16  # 使用 bf16
        
        # 加载基础模型
        logger.info("正在加载模型...")
        
        # 设置device_map
        model_config["device_map"] = None if args.deepspeed else "auto"
            
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
            
        # 如果使用 LoRA，添加 LoRA 配置
        if args.training_mode == "lora":
            # 冻结所有参数
            for param in model.parameters():
                param.requires_grad = False
            
            target_modules = args.lora_target_modules.split(",")
            logger.info(f"LoRA目标模块: {target_modules}")
            
            
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
            )
            
            # 应用 LoRA 配置
            model = get_peft_model(model, lora_config)
            
            # 检查LoRA参数是否正确设置
            trainable_params = []
            all_param_size = 0
            trainable_param_size = 0
            
            for name, param in model.named_parameters():
                all_param_size += param.numel()
                if param.requires_grad:
                    trainable_params.append(name)
                    trainable_param_size += param.numel()
                    
            logger.info(f"可训练参数量: {trainable_param_size:,}")
            logger.info(f"总参数量: {all_param_size:,}")
            logger.info(f"可训练参数占比: {trainable_param_size/all_param_size:.2%}")
            
        # 加载数据集
        logger.info("正在加载数据集...")
        try:
            import json
            from datasets import Dataset
            
            # 加载JSON数组格式的文件
            logger.info(f"从文件加载数据: {args.dataset_name}")
            data = []
            with open(args.dataset_name, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:  # 跳过空行
                            continue
                        item = json.loads(line)
                        if isinstance(item, dict):
                            data.append(item)
                        else:
                            logger.warning(f"第{line_num}行数据格式错误，期望字典类型，实际得到 {type(item)}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败：{str(e)}")
                    except Exception as e:
                        logger.warning(f"第{line_num}行处理失败：{str(e)}")
                        
            if not data:
                raise ValueError("没有成功加载任何数据")
                
            logger.info(f"成功加载 {len(data)} 条数据")
                
            # 标准化数据
            normalized_data = normalize_data(data)
            logger.info(f"数据标准化完成，处理后数据量: {len(normalized_data)}")
                
            # 转换为Dataset格式
            dataset = Dataset.from_list(normalized_data)
            
            # 打印原始数据集信息
            logger.info(f"\n数据集信息:")
            logger.info(f"训练集大小: {len(dataset)} 条")
            logger.info("\n示例数据:")
            for i in range(min(3, len(dataset))):
                logger.info(f"\n示例 {i+1}:")
                for key, value in dataset[i].items():
                    logger.info(f"{key}: {value}")
                logger.info("-" * 50)
            
            # 预处理数据集
            logger.info("\n正在预处理数据集...")
            conversations = dataset.map(
                preprocess_chat_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="处理对话数据",
            )
            
            # 打印预处理后的数据
            logger.info("\n预处理后的数据示例:")
            for i in range(min(3, len(conversations))):
                logger.info(f"\n对话 {i+1}:")
                logger.info(conversations[i]["conversations"])
                logger.info("-" * 50)
            
            # 使用tokenizer处理文本
            def tokenize_function(examples):
                inputs = tokenizer(
                    examples["conversations"],
                    truncation=True,
                    max_length=args.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                inputs["labels"] = inputs["input_ids"].clone()
                inputs["labels"][inputs["input_ids"] == tokenizer.pad_token_id] = -100
                return inputs
                
            train_dataset = conversations.map(
                tokenize_function,
                batched=True,
                remove_columns=conversations.column_names,
                desc="tokenizing",
            )
            
        except Exception as e:
            logger.error(f"数据加载出错: {str(e)}")
            raise
        
        # 训练配置
        logger.info("训练配置:")
        logger.info(f"- 批次大小: {args.per_device_train_batch_size}")
        logger.info(f"- 梯度累积步数: {args.gradient_accumulation_steps}")
        logger.info(f"- 有效批次大小: {args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size}")
        logger.info(f"- 学习率: {args.learning_rate}")
        logger.info(f"- 训练轮数: {args.num_train_epochs}")
        logger.info(f"- 最大序列长度: {args.max_seq_length}")
        
        # 创建训练器
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,  # SFTTrainer内部会处理tokenizer的弃用警告
            data_collator=None,  # 让trainer自动创建数据整理器
        )
        
        # 开始训练
        logger.info("开始训练...")
        try:
            # 检查是否存在检查点
            last_checkpoint = None
            if os.path.exists(args.output_dir):
                checkpoints = [
                    folder 
                    for folder in os.listdir(args.output_dir)
                    if folder.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, folder))
                ]
                if checkpoints:
                    # 按检查点编号排序
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    last_checkpoint = os.path.join(args.output_dir, checkpoints[-1])
                    # 验证检查点完整性
                    if os.path.exists(os.path.join(last_checkpoint, "trainer_state.json")):
                        logger.info(f"发现有效检查点: {last_checkpoint}")
                    else:
                        logger.warning(f"检查点 {last_checkpoint} 不完整，将从头开始训练")
                        last_checkpoint = None
            
            if last_checkpoint is None:
                logger.info("未发现有效检查点，从头开始训练")
            
            # 开始训练
            trainer.train(resume_from_checkpoint=last_checkpoint)
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU内存不足，请减小batch_size或使用gradient_accumulation")
            raise
        except Exception as e:
            logger.error(f"训练过程出错: {str(e)}")
            if trainer.is_world_process_zero():
                trainer.save_state()
            raise
            
        # 保存最终模型
        logger.info("正在保存模型...")
        save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_path, exist_ok=True)
        
        if args.training_mode == "lora":
            # 确保所有进程同步
            logger.info("等待所有进程同步...")
            trainer.accelerator.wait_for_everyone()
            
            # 获取解包后的模型
            logger.info("正在解包模型...")
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
            
            # 在主进程上保存
            if trainer.accelerator.is_main_process:
                logger.info(f"正在保存LoRA模型到: {save_path}")
                try:
                    unwrapped_model.save_pretrained(
                        save_path,
                        save_function=trainer.accelerator.save,
                    )
                    logger.info("LoRA模型保存成功！")
                    
                    # 验证保存的文件
                    saved_files = os.listdir(save_path)
                    logger.info(f"保存的文件列表: {saved_files}")
                    
                    # 检查权重文件（支持.bin和.safetensors格式）
                    adapter_model_path = os.path.join(save_path, "adapter_model.safetensors")
                    if not os.path.exists(adapter_model_path):
                        adapter_model_path = os.path.join(save_path, "adapter_model.bin")
                    
                    if os.path.exists(adapter_model_path):
                        logger.info(f"LoRA权重文件 ({os.path.basename(adapter_model_path)}) 大小: {os.path.getsize(adapter_model_path) / 1024 / 1024:.2f}MB")
                    else:
                        logger.error("错误：未找到LoRA权重文件！(检查了.safetensors和.bin格式)")
                        
                except Exception as e:
                    logger.error(f"保存模型时出错: {str(e)}")
                    raise
                    
                trainer.tokenizer.save_pretrained(save_path)
        else:
            # 全参数微调模型的保存逻辑
            logger.info("正在保存全参数微调模型...")
            
            # 确保所有进程同步
            trainer.accelerator.wait_for_everyone()
            
            # 获取解包后的模型
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
            
            # 在主进程上保存
            if trainer.accelerator.is_main_process:
                # 保存模型
                unwrapped_model.save_pretrained(
                    save_path,
                    safe_serialization=True,  # 使用safetensors格式保存
                    save_function=trainer.accelerator.save,
                )
                
                # 保存分词器和配置
                trainer.tokenizer.save_pretrained(save_path)
                
                # 验证保存的文件
                config_path = os.path.join(save_path, "config.json")
                if os.path.exists(config_path):
                    logger.info("已保存config.json")
                else:
                    logger.error("config.json未保存！")
                
                # 检查权重文件
                if any(f.endswith(".safetensors") or f.endswith(".bin") for f in os.listdir(save_path)):
                    logger.info("已保存模型权重文件")
                else:
                    logger.error("未找到模型权重文件！")
                
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}")
        raise
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