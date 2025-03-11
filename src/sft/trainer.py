"""
SFT训练器模块
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollator,
)
from datasets import Dataset
from peft import PeftModel
from sklearn.metrics import f1_score, recall_score

logger = logging.getLogger(__name__)

class SFTTrainer(Trainer):
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
        # 初始化损失记录
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        
        # 如果使用DeepSpeed，移除wandb回调
        if args.deepspeed:
            callbacks = [] if callbacks is None else callbacks
            callbacks = [cb for cb in callbacks if not str(cb.__class__).endswith("WandBCallback")]
            args.report_to = [r for r in args.report_to if r != "wandb"]
        
        # 验证训练数据集
        if train_dataset is not None:
            required_columns = ["input_ids", "labels"]
            missing_columns = [col for col in required_columns if col not in train_dataset.column_names]
            if missing_columns:
                raise ValueError(f"训练数据集必须包含 'input_ids' 和 'labels' 列")
        
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
        
        self.tokenizer = tokenizer
        
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
            if isinstance(model, PeftModel):
                try:
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    all_params = sum(p.numel() for p in model.parameters())
                    wandb.config.update({
                        "trainable_params": trainable_params,
                        "total_params": all_params,
                        "trainable_ratio": trainable_params / all_params,
                    })
                except Exception:
                    pass

    def log(self, logs: Dict[str, float], iterator: Optional[Any] = None) -> None:
        """重写日志记录方法，添加损失记录"""
        if self.state.global_step:  # 确保不是初始步骤
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.steps.append(self.state.global_step)
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                
            # 在每个检查点保存损失图
            if self.is_world_process_zero() and self.state.global_step % self.args.save_steps == 0:
                self.save_loss_plots()
                
        super().log(logs, iterator)

    def save_loss_plots(self):
        """保存损失图表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.args.output_dir, "loss_plots")
        os.makedirs(save_dir, exist_ok=True)
        
        # 清理matplotlib
        plt.clf()
        
        # 绘制训练损失
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.train_losses, label='Training Loss')
        if self.eval_losses:
            eval_steps = [s for s in self.steps if s % self.args.eval_steps == 0][:len(self.eval_losses)]
            plt.plot(eval_steps, self.eval_losses, label='Evaluation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        plot_path = os.path.join(save_dir, f'loss_plot_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        
        # 如果使用wandb，上传图片
        if self.is_world_process_zero() and not os.getenv("WANDB_DISABLED", "false").lower() == "true":
            wandb.log({
                "loss_plot": wandb.Image(plot_path),
                "current_train_loss": self.train_losses[-1] if self.train_losses else None,
                "current_eval_loss": self.eval_losses[-1] if self.eval_losses else None,
            })

    def __del__(self):
        """清理函数，确保正确关闭wandb和保存最终的损失图"""
        try:
            if hasattr(self, 'args') and self.args is not None:
                self.save_loss_plots()
            if hasattr(self, 'is_world_process_zero') and \
               self.is_world_process_zero() and \
               wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    def get_train_dataloader(self):
        """获取训练数据加载器"""
        return super().get_train_dataloader()
        
    def get_eval_dataloader(self, eval_dataset=None):
        """获取评估数据加载器"""
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return super().get_eval_dataloader(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算损失，支持返回模型输出"""
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
        
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        **kwargs
    ):
        """评估循环，计算评估指标"""
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 调用父类的评估循环
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            **kwargs
        )
        
        # 如果只计算损失，直接返回
        if prediction_loss_only:
            return eval_output
            
        metrics = {}
        
        # 计算perplexity
        if eval_output.predictions is not None:
            metrics["perplexity"] = torch.exp(torch.tensor(eval_output.metrics["eval_loss"])).item()
            
        # 更新指标
        eval_output.metrics.update(metrics)
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return eval_output 