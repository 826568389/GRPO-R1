"""
GRPO-R1的主要训练脚本
实现了模型训练的主要流程，包括数据加载、模型初始化、训练循环等
"""

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer


from configs import GRPOConfig
from rewards import REWARD_FUNCS_REGISTRY
from utils.callbacks import get_callbacks
from grpo_trainer import GRPOTrainerExt
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    GRPO训练脚本的参数配置类
    继承自TRL的ScriptArguments，添加了奖励函数相关的配置
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": f"奖励函数列表，可选值: {', '.join(REWARD_FUNCS_REGISTRY.keys())}"
        },
    )


# 系统提示词，用于构建对话格式
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    """
    主训练函数
    
    参数:
        script_args: 脚本参数，包含数据集配置、奖励函数等
        training_args: 训练参数，包含学习率、批次大小等
        model_args: 模型参数，包含模型路径、类型等
    """
    # 设置随机种子以确保可重复性
    set_seed(training_args.seed)

    # 设置日志配置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 在每个进程上记录简要信息
    logger.warning(
        f"进程排名: {training_args.local_rank}, 设备: {training_args.device}, GPU数量: {training_args.n_gpu}"
        + f" 分布式训练: {bool(training_args.local_rank != -1)}, 16位训练: {training_args.fp16}"
    )
    logger.info(f"模型参数 {model_args}")
    logger.info(f"脚本参数 {script_args}")
    logger.info(f"数据参数 {training_args}")

    # 检查最新的检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"检测到检查点，从 {last_checkpoint} 恢复训练。")

    # 加载数据集
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # 获取奖励函数
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # 将数据格式化为对话格式
    def make_conversation(example):
        """将示例转换为对话格式"""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # 初始化模型参数
    logger.info("*** 初始化模型参数 ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    training_args.gradient_checkpointing = True
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        load_in_4bit=False, 
        **model_kwargs
    )

    print(model_args.model_name_or_path,)

    # 初始化GRPO-R1训练器
    trainer = GRPOTrainerExt(
        model = model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        callbacks=get_callbacks(training_args, model_args),
    )

    # 开始训练
    logger.info("*** 开始训练 ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 保存模型和创建模型卡片
    logger.info("*** 保存模型 ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"模型已保存到 {training_args.output_dir}")

    # 在主进程上保存其他内容
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["GRPO-R1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # 恢复k,v缓存以加速推理
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    # 解析命令行参数并启动训练
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
