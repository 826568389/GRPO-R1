#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=28500

# 获取项目根目录
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# 添加项目根目录到PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 设置wandb配置
export WANDB_PROJECT="sft-training-full"
export WANDB_WATCH="gradients"
export WANDB_LOG_MODEL="checkpoint"
export WANDB_DISABLED="false"

# 执行训练命令
deepspeed --num_gpus 4 ${PROJECT_ROOT}/src/sft/train.py \
    --model_name_or_path "/data/staryea/aigc_model/Qwen2.5-0.5B-Instruct" \
    --dataset_name "/data/staryea/DeepSeek/dataset/BelleGroup/train_0.2k_CN" \
    --output_dir "output/Qwen2.5-0.5B-Instruct-sft-full" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --max_seq_length 2048 \
    --save_strategy "steps" \
    --save_steps 500 \
    --logging_steps 100 \
    --use_gradient_checkpointing true \
    --fp16 true \
    --deepspeed "${PROJECT_ROOT}/config/sft_zero3.json" \
    --report_to "wandb" \
    --full_finetune true \
    --lora_rank 0 \
    --use_peft false 