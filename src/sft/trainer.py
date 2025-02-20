"""
SFT训练器模块
实现了基于DeepSpeed ZeRO的全参数微调训练器
"""

import os
import logging
from typing import Optional, Dict, Any
import time

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
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # 初始化训练状态，只保留核心指标
        self.train_stats = {
            "train/loss": [],
            "train/learning_rate": [],
            "train/epoch": [],
            "train/step": [],
            "train/perplexity": [],
            "train/grad_norm": [],
            "train/progress": [],  # 训练进度百分比
        }
        
        # 初始化wandb
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
            
            try:
                # 记录模型参数量
                named_parameters = list(model.named_parameters())
                total_params = sum(p.numel() for name, p in named_parameters)
                trainable_params = sum(p.numel() for name, p in named_parameters if p.requires_grad)
                
                if total_params > 0:
                    wandb.config.update({
                        "total_params": total_params,
                        "trainable_params": trainable_params,
                        "trainable_ratio": trainable_params / total_params
                    })
                    logger.info(f"模型总参数量: {total_params:,}")
                    logger.info(f"可训练参数量: {trainable_params:,}")
                    logger.info(f"可训练参数比例: {trainable_params/total_params:.2%}")
                else:
                    logger.warning("无法获取模型参数量信息")
            except Exception as e:
                logger.warning(f"记录模型参数量时出错: {str(e)}")
    
    def log_metrics(self, split, metrics, epoch=None):
        """重写日志记录方法，确保所有指标都被正确记录"""
        if self.is_world_process_zero():
            try:
                # 添加前缀
                metrics = {f"{split}/{k}" if not k.startswith(split) else k: v 
                          for k, v in metrics.items()}
                
                # 记录到wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=self.state.global_step)
                
                # 保存到训练状态
                for key, value in metrics.items():
                    if key in self.train_stats:
                        self.train_stats[key].append(value)
                
            except Exception as e:
                logger.warning(f"记录指标时出错: {str(e)}")
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """重写训练步骤，添加指标记录"""
        # 执行原始的训练步骤
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # 记录训练指标
        if self.is_world_process_zero() and self.state.global_step % self.args.logging_steps == 0:
            try:
                # 计算训练进度
                if self.args.max_steps > 0:
                    progress = self.state.global_step / self.args.max_steps
                else:
                    progress = self.state.epoch
                
                # 基本训练指标
                metrics = {
                    "train/loss": loss.item(),
                    "train/epoch": self.state.epoch,
                    "train/step": self.state.global_step,
                    "train/learning_rate": self.get_current_lr(),
                    "train/progress": progress * 100,  # 转换为百分比
                }
                
                # 计算perplexity
                if not torch.isnan(loss) and not torch.isinf(loss):
                    perplexity = torch.exp(loss)
                    if not torch.isnan(perplexity) and not torch.isinf(perplexity):
                        metrics["train/perplexity"] = perplexity.item()
                
                # 计算梯度范数
                if hasattr(self, "optimizer"):
                    grad_norm = self.get_grad_norm()
                    if grad_norm is not None:
                        metrics["train/grad_norm"] = grad_norm
                
                # 记录所有指标
                self.log_metrics("train", metrics)
                
            except Exception as e:
                logger.warning(f"记录训练指标时出错: {str(e)}")
                logger.debug(f"错误详情: {str(e)}", exc_info=True)
        
        return loss
    
    def get_current_lr(self) -> float:
        """获取当前学习率"""
        try:
            # 尝试从优化器获取学习率
            if self.optimizer is not None:
                return self.optimizer.param_groups[0]["lr"]
            # 尝试从调度器获取学习率
            elif self.lr_scheduler is not None:
                return self.lr_scheduler.get_last_lr()[0]
            # 如果都没有，返回配置的学习率
            else:
                return self.args.learning_rate
        except Exception as e:
            logger.warning(f"获取学习率时出错: {str(e)}")
            return self.args.learning_rate
    
    def get_grad_norm(self) -> Optional[float]:
        """计算梯度范数"""
        if not hasattr(self, "optimizer"):
            return None
        
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _prepare_deepspeed(self):
        """准备DeepSpeed配置"""
        deepspeed_config = {
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
                "enabled": self.args.deepspeed_config.fp16,
            },
            "zero_optimization": {
                "stage": self.args.deepspeed_config.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.args.deepspeed_config.offload_optimizer else "none"
                },
                "offload_param": {
                    "device": "cpu" if self.args.deepspeed_config.offload_param else "none"
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
            },
            "gradient_clipping": self.args.deepspeed_config.gradient_clipping,
            "steps_per_print": self.args.logging_steps,
        }
        return deepspeed_config
        
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        trial: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        开始训练
        
        参数:
            resume_from_checkpoint: 从检查点恢复训练的路径
            trial: optuna超参搜索的trial对象
            **kwargs: 其他参数
        """
        try:
            # 记录训练开始时间
            self._train_start_time = time.time()
            
            # 如果使用 DeepSpeed，移除 wandb 回调以避免冲突
            if self.args.deepspeed:
                # 保存原始回调
                original_callbacks = self.callback_handler.callbacks
                # 过滤掉 WandbCallback
                filtered_callbacks = [
                    callback for callback in original_callbacks 
                    if not callback.__class__.__name__ == "WandbCallback"
                ]
                self.callback_handler.callbacks = filtered_callbacks
            
            # 准备DeepSpeed
            if self.args.deepspeed:
                logger.info("正在初始化 DeepSpeed...")
                deepspeed_config = self._prepare_deepspeed()
                
                # 确保 DeepSpeed 配置正确
                if "zero_optimization" in deepspeed_config:
                    logger.info(f"使用 ZeRO-{deepspeed_config['zero_optimization']['stage']} 优化")
                    if deepspeed_config['zero_optimization']['stage'] == 3:
                        logger.info("启用 ZeRO-3 优化")
                        # 对于ZeRO-3，确保启用了16位模型保存
                        deepspeed_config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
                
                # 更新DeepSpeed配置
                self.args.deepspeed = deepspeed_config
                
            # 调用父类的训练方法
            result = super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                **kwargs,
            )
            
            # 保存最终模型
            if self.is_world_process_zero():
                logger.info("正在保存最终模型...")
                # 保存模型
                if self.args.training_mode == "full":
                    # 全量模型保存
                    self.save_model(self.args.output_dir)
                elif self.args.training_mode in ["lora", "qlora"]:
                    # LoRA模型保存
                    self.model.save_pretrained(self.args.output_dir)
                    # 保存分词器
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(self.args.output_dir)
                    logger.info(f"LoRA权重已保存到: {self.args.output_dir}")
                
                # 保存训练参数和统计信息
                self.save_state()
                self._save_training_stats()
            
            # 记录最终的训练指标
            if self.is_world_process_zero() and wandb.run is not None:
                # 计算训练时间（秒）
                total_train_time = time.time() - self._train_start_time
                
                final_metrics = {
                    "train/final_loss": result.training_loss,
                    "train/total_steps": result.global_step,
                    "train/total_time": total_train_time,
                }
                
                # 如果有速度指标，也记录它们
                if hasattr(result, "metrics"):
                    if "train_samples_per_second" in result.metrics:
                        final_metrics["train/samples_per_second"] = result.metrics["train_samples_per_second"]
                    if "train_steps_per_second" in result.metrics:
                        final_metrics["train/steps_per_second"] = result.metrics["train_steps_per_second"]
                
                wandb.log(final_metrics)
            
            return result
            
        finally:
            # 训练结束，确保关闭wandb
            if self.is_world_process_zero() and wandb.run is not None:
                wandb.finish()
            
            # 如果之前移除了回调，现在恢复它们
            if self.args.deepspeed and hasattr(self, "original_callbacks"):
                self.callback_handler.callbacks = original_callbacks
    
    def _save_training_stats(self):
        """保存训练统计信息"""
        import json
        import os
        
        stats_path = os.path.join(self.args.output_dir, "training_stats.json")
        try:
            # 将训练统计信息保存为JSON
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(self.train_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"训练统计信息已保存到: {stats_path}")
        except Exception as e:
            logger.warning(f"保存训练统计信息时出错: {str(e)}")
    
    def save_model(self, output_dir: str, _internal_call: bool = False):
        """保存完整模型
        
        Args:
            output_dir: 输出目录
            _internal_call: 是否是内部调用（用于与父类兼容）
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 只在非内部调用时输出日志
        if not _internal_call and self.is_world_process_zero():
            logger.info(f"保存模型到: {output_dir}")
        
        if self.args.deepspeed:
            # 使用DeepSpeed的保存方法
            if hasattr(self.model, "module") and hasattr(self.model, "save_pretrained"):
                # 如果模型被DeepSpeed包装，使用save_pretrained方法
                if self.is_world_process_zero():
                    self.model.module.save_pretrained(output_dir)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(output_dir)
            else:
                # 只在非内部调用时输出警告
                if not _internal_call and self.is_world_process_zero():
                    logger.warning("模型未被DeepSpeed正确初始化，使用普通方式保存")
                if self.is_world_process_zero():
                    self.model.save_pretrained(output_dir)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(output_dir)
        else:
            # 使用普通的保存方法
            if self.is_world_process_zero():
                self.model.save_pretrained(output_dir)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
        
        # 如果不是内部调用，则保存训练状态
        if not _internal_call and self.is_world_process_zero():
            self.save_state()
            self._save_training_stats() 