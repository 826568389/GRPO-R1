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
    
    def log_metrics(self, split, metrics, epoch=None):
        """记录训练指标到wandb"""
        if self.is_world_process_zero() and wandb.run is not None:
            # 添加前缀
            metrics = {f"{split}/{k}" if not k.startswith(split) else k: v 
                      for k, v in metrics.items()}
            # 记录到wandb
            wandb.log(metrics, step=self.state.global_step)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """执行训练步骤并记录指标"""
        # 执行训练
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # 记录训练指标
        if self.is_world_process_zero() and self.state.global_step % self.args.logging_steps == 0:
            try:
                # 计算训练进度
                progress = (self.state.global_step / self.args.max_steps * 100) if self.args.max_steps > 0 else (self.state.epoch * 100)
                
                # 记录基本指标
                metrics = {
                    "train/loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train/epoch": self.state.epoch,
                    "train/step": self.state.global_step,
                    "train/progress": progress,
                }
                
                # 记录perplexity
                if not torch.isnan(loss) and not torch.isinf(loss):
                    metrics["train/perplexity"] = torch.exp(loss).item()
                
                self.log_metrics("train", metrics)
                
            except Exception as e:
                logger.warning(f"记录训练指标时出错: {str(e)}")
        
        return loss
    
    def _prepare_deepspeed(self):
        """准备DeepSpeed配置"""
        return {
            "train_micro_batch_size_per_gpu": self.args.per_device_train_batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.args.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.args.learning_rate,
                    "warmup_num_steps": self.args.warmup_steps,
                    "total_num_steps": self.args.max_steps,
                }
            },
            "fp16": {
                "enabled": True,
                "auto_cast": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
            },
            "gradient_clipping": 1.0,
            "steps_per_print": self.args.logging_steps,
        }
    
    def get_train_dataloader(self):
        """获取训练数据加载器，添加数据验证"""
        dataloader = super().get_train_dataloader()
        if self.is_world_process_zero():
            # 检查第一个批次
            batch = next(iter(dataloader))
            logger.info(f"训练数据批次格式: {batch.keys()}")
            logger.info(f"输入形状: {batch['input_ids'].shape}")
            logger.info(f"标签形状: {batch['labels'].shape}")
        return dataloader 