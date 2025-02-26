#!/bin/bash

# 设置环境变量
export MASTER_PORT=28500

# 获取项目根目录
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# 添加项目根目录到PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 设置wandb配置
export WANDB_PROJECT="sft-training-lora"
export WANDB_WATCH="gradients"
export WANDB_LOG_MODEL="checkpoint"
export WANDB_DISABLED="false"

# 执行训练命令
deepspeed --include localhost:0,1,2,3 ${PROJECT_ROOT}/src/sft/train.py \
    --model_name_or_path "/data/staryea/aigc_model/Qwen2.5-7B-Instruct" \
    --dataset_name "/data/staryea/DeepSeek/dataset/CVE_QA/cve_9.jsonl" \
    --output_dir "output/Qwen2.5-7B-Instruct-sft-lora-0224" \
    --num_train_epochs 50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --max_seq_length 2048 \
    --save_strategy "steps" \
    --save_steps 10 \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.0 \
    --use_gradient_checkpointing true \
    --bf16 true \
    --deepspeed "${PROJECT_ROOT}/config/sft_zero3.json" \
    --training_mode "lora" \
    --report_to "wandb" \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --seed 42 \
    --weight_decay 0.0 \
    --lora_target_modules "q_proj,v_proj,o_proj,k_proj"
