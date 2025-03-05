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

# 设置CUDA内存分配器配置
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 执行训练命令
deepspeed --include localhost:0,1,2,3 ${PROJECT_ROOT}/src/sft/train.py \
    --model_name_or_path "/data/staryea/aigc_model/Qwen2.5-3B-Instruct" \
    --dataset_name "/data/staryea/DeepSeek/dataset/BusinessData/sft_data/opt_sft_v6.jsonl" \
    --output_dir "output/Qwen2.5-3B-Instruct-Business-lora-03051" \
    --num_train_epochs 550 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --max_seq_length 2048 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --use_gradient_checkpointing true \
    --bf16 true \
    --deepspeed "${PROJECT_ROOT}/config/sft_zero2.json" \
    --training_mode "lora" \
    --report_to "wandb" \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --seed 42 \
    --weight_decay 0.01 \
    --lora_target_modules "q_proj,v_proj,o_proj,k_proj,down_proj,up_proj,gate_proj" 