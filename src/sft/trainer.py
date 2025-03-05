"""
SFT训练器模块
实现了基于DeepSpeed ZeRO的全参数微调训练器
"""

import os
import logging
from typing import Optional, Dict, Any

import torch
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollator,
)
from datasets import Dataset
from peft import PeftModel

logger = logging.getLogger(__name__)


class SFTTrainer(Trainer):
    """
    基于DeepSpeed ZeRO的SFT训练器
    继承自transformers的Trainer类，添加了DeepSpeed相关的功能
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        data_collator: Optional[DataCollator] = None,
        callbacks: list = None,
    ):
        # 如果使用DeepSpeed，移除wandb回调
        if args.deepspeed:
            callbacks = [] if callbacks is None else callbacks
            callbacks = [cb for cb in callbacks if not str(cb.__class__).endswith("WandBCallback")]
            # 禁用wandb报告
            args.report_to = [r for r in args.report_to if r != "wandb"]
        
        # 确保数据集格式正确
        if train_dataset is not None:
            if not all(col in train_dataset.features for col in ["input_ids", "labels"]):
                raise ValueError("训练数据集必须包含 'input_ids' 和 'labels' 列")
        
        if eval_dataset is not None:
            if not all(col in eval_dataset.features for col in ["input_ids", "labels"]):
                raise ValueError("验证数据集必须包含 'input_ids' 和 'labels' 列")
        
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # 初始化wandb（仅在主进程且未禁用时）
        if self.is_world_process_zero() and not os.getenv("WANDB_DISABLED", "false").lower() == "true":
            run_name = f"sft-{args.model_name_or_path.split('/')[-1]}"
            if hasattr(args, "run_name") and args.run_name:
                run_name = args.run_name

            wandb.init(
                project=os.getenv("WANDB_PROJECT", "sft-training"),
                name=run_name,
                config={
                    "model": args.model_name_or_path,
                    "batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size,
                    "learning_rate": args.learning_rate,
                    "epochs": args.num_train_epochs,
                    "max_steps": args.max_steps,
                    "warmup_steps": args.warmup_steps,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "max_seq_length": args.max_seq_length,
                }
            )
            
            # 记录模型参数量
            try:
                if isinstance(model, PeftModel):
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    all_params = sum(p.numel() for p in model.parameters())
                    wandb.config.update({
                        "trainable_params": trainable_params,
                        "total_params": all_params,
                        "trainable_ratio": trainable_params / all_params,
                    })
                
            except Exception as e:
                logger.warning(f"记录模型参数量时出错: {str(e)}")
    
    def __del__(self):
        """
        清理函数，确保正确关闭wandb
        """
        try:
            import wandb
            if hasattr(self, 'args') and self.args is not None and \
               hasattr(self, 'is_world_process_zero') and \
               self.is_world_process_zero() and \
               wandb.run is not None:
                wandb.finish()
        except Exception:
            pass  # 忽略清理过程中的任何错误
    


        """获取训练数据加载器，添加数据验证"""
        try:
            dataloader = super().get_train_dataloader()
            
            if self.is_world_process_zero():
                # 检查第一个批次
                batch = next(iter(dataloader))
                logger.info("训练数据批次信息:")
                logger.info(f"- 批次键: {batch.keys()}")
                for key, value in batch.items():
                    if hasattr(value, 'shape'):
                        logger.info(f"- {key} 形状: {value.shape}")
                        if torch.is_tensor(value):
                            logger.info(f"  - 数据类型: {value.dtype}")
                            logger.info(f"  - 设备: {value.device}")
                            if value.numel() > 0:
                                logger.info(f"  - 值范围: [{value.min()}, {value.max()}]")
                
                # 验证批次大小
                if hasattr(self.args, 'per_device_train_batch_size'):
                    expected_batch_size = self.args.per_device_train_batch_size
                    actual_batch_size = batch['input_ids'].size(0)
                    if actual_batch_size != expected_batch_size:
                        logger.warning(f"实际批次大小 ({actual_batch_size}) 与配置的批次大小 ({expected_batch_size}) 不匹配")
                
            return dataloader
            
        except Exception as e:
            logger.error(f"获取训练数据加载器失败: {str(e)}")
            logger.debug("错误详情:", exc_info=True)
            raise 