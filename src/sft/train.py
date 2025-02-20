"""
SFT训练入口脚本
实现了数据加载和训练流程
"""

import os
import sys
import signal
import logging
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    AutoConfig,
)

from configs import SFTArguments
from trainer import SFTTrainer

logger = logging.getLogger(__name__)

# 全局变量用于追踪训练状态
_training_finished = False
_save_in_progress = False

def signal_handler(signum, frame):
    """处理中断信号"""
    global _training_finished, _save_in_progress
    
    if _save_in_progress:
        logger.warning("正在保存模型，请等待...")
        return
        
    if not _training_finished:
        logger.info("接收到中断信号，正在安全退出...")
        _training_finished = True
        sys.exit(0)

def train():
    global _training_finished, _save_in_progress
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 解析命令行参数
        parser = HfArgumentParser(SFTArguments)
        args = parser.parse_args_into_dataclasses()[0]
        
        # 设置随机种子
        set_seed(args.seed)
        
        # 设置运行名称
        if not args.run_name:
            model_name = args.model_name_or_path.split('/')[-1]
            args.run_name = f"sft-{model_name}-full-{args.learning_rate}"
        
        # 加载tokenizer
        logger.info("正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
        
        # 准备模型配置
        logger.info("正在准备模型配置...")
        model_config = {
            "use_cache": False if args.use_gradient_checkpointing else True,
            "trust_remote_code": True,
        }
        
        # 设置数据类型
        if args.bf16:
            model_config["torch_dtype"] = torch.bfloat16
        elif args.fp16:
            model_config["torch_dtype"] = torch.float16
        
        # 加载基础模型
        logger.info("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_config
        )
        
        # 加载数据集
        logger.info("正在加载数据集...")
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
        logger.info("正在预处理数据集...")
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
        logger.info("开始训练...")
        trainer.train()
        _training_finished = True
        
        # 保存最终模型
        logger.info("正在保存模型...")
        _save_in_progress = True
        save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_path, exist_ok=True)
        
        # 确保模型在保存前处于eval模式
        logger.info("将模型设置为eval模式...")
        trainer.model.eval()
        
        # 如果使用了DeepSpeed，需要特殊处理
        if trainer.is_deepspeed_enabled:
            logger.info("检测到DeepSpeed，使用特殊保存流程...")
            
            try:
                # 确保所有进程同步
                logger.info("等待所有进程同步...")
                trainer.accelerator.wait_for_everyone()
                
                # 获取DeepSpeed引擎
                ds_engine = trainer.deepspeed
                
                if ds_engine is None:
                    raise ValueError("DeepSpeed引擎未初始化")
                
                # 保存checkpoint
                logger.info("保存DeepSpeed checkpoint...")
                checkpoint_path = os.path.join(save_path, "ds_checkpoint")
                ds_engine.save_checkpoint(checkpoint_path)
                
                if trainer.is_world_process_zero():
                    try:
                        # 使用DeepSpeed的状态字典加载器
                        logger.info("正在合并分片参数...")
                        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
                        
                        # 获取HF模型配置
                        config = AutoConfig.from_pretrained(
                            args.model_name_or_path,
                            trust_remote_code=True
                        )
                        
                        # 创建新的模型实例
                        logger.info("创建新的模型实例...")
                        # 获取正确的模型类
                        if hasattr(trainer.model, "module"):
                            model_class = type(trainer.model.module)
                        else:
                            model_class = type(trainer.model)
                            
                        logger.info(f"使用模型类: {model_class.__name__}")
                        new_model = model_class(config)
                        
                        # 加载状态字典
                        logger.info("加载合并后的参数...")
                        # 处理lm_head权重
                        if "model.embed_tokens.weight" in state_dict:
                            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
                            logger.info("从embed_tokens复制权重到lm_head")
                            
                        missing_keys, unexpected_keys = new_model.load_state_dict(state_dict, strict=False)
                        if missing_keys:
                            logger.warning(f"加载状态字典时缺少的键: {missing_keys}")
                        if unexpected_keys:
                            logger.warning(f"加载状态字典时未预期的键: {unexpected_keys}")
                            
                        # 确保lm_head权重正确设置
                        if hasattr(new_model, "lm_head") and hasattr(new_model, "model"):
                            if hasattr(new_model.model, "embed_tokens"):
                                logger.info("设置lm_head权重与embed_tokens共享")
                                new_model.lm_head.weight = new_model.model.embed_tokens.weight
                        
                        # 保存完整模型
                        logger.info("保存完整模型...")
                        new_model.save_pretrained(
                            save_path,
                            safe_serialization=True,
                            max_shard_size="10GB"
                        )
                        
                        # 保存tokenizer和配置
                        trainer.tokenizer.save_pretrained(save_path)
                        
                        # 清理临时checkpoint
                        logger.info("清理临时文件...")
                        import shutil
                        shutil.rmtree(checkpoint_path, ignore_errors=True)
                        
                        # 验证保存的文件
                        logger.info("验证保存的文件...")
                        saved_files = os.listdir(save_path)
                        total_size = 0
                        logger.info("已保存的文件:")
                        for file in saved_files:
                            file_path = os.path.join(save_path, file)
                            file_size = os.path.getsize(file_path)
                            total_size += file_size
                            logger.info(f"  - {file} ({file_size/1024/1024:.2f}MB)")
                        logger.info(f"总文件大小: {total_size/1024/1024:.2f}MB")
                        
                        if total_size < 100 * 1024 * 1024:
                            raise Exception("保存的模型文件过小，可能未正确保存")
                            
                    except Exception as e:
                        logger.error(f"保存模型时出错: {str(e)}")
                        raise
                
                # 最终同步
                trainer.accelerator.wait_for_everyone()
                
            finally:
                _save_in_progress = False
                logger.info("模型保存流程完成")
                
        else:
            # 普通模式保存
            try:
                logger.info("使用标准模式保存模型...")
                trainer.save_model(save_path)
                logger.info("模型保存成功")
                
                if trainer.is_world_process_zero():
                    # 验证保存的文件
                    logger.info("验证保存的文件...")
                    saved_files = os.listdir(save_path)
                    total_size = 0
                    logger.info("已保存的文件:")
                    for file in saved_files:
                        file_path = os.path.join(save_path, file)
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        logger.info(f"  - {file} ({file_size/1024/1024:.2f}MB)")
                    logger.info(f"总文件大小: {total_size/1024/1024:.2f}MB")
                    
                    if total_size < 100 * 1024 * 1024:
                        raise Exception("保存的模型文件过小，可能未正确保存")
                        
            except Exception as e:
                logger.error(f"保存模型时出错: {str(e)}")
                raise
            finally:
                _save_in_progress = False
                
    except KeyboardInterrupt:
        logger.info("接收到用户中断，正在安全退出...")
        sys.exit(0)
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