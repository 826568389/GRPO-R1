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
    --model_name_or_path "/data/staryea/aigc_model/Qwen2.5-7B-Instruct" \
    --dataset_name "/data/staryea/DeepSeek/dataset/BusinessData/sft_data/sft_data.jsonl" \
    --output_dir "output/Qwen2.5-7B-Instruct-Business-lora" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --max_seq_length 2048 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 2 \
    --eval_steps 10 \
    --evaluation_strategy "steps" \
    --max_grad_norm 0.3 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_ratio 0.1 \
    --use_gradient_checkpointing true \
    --bf16 true \
    --deepspeed "${PROJECT_ROOT}/config/sft_zero2.json" \
    --training_mode "lora" \
    --report_to "wandb" \
    --lora_r 32 \
    --lora_alpha 128 \
    --lora_dropout 0.2 \
    --seed 42 \
    --weight_decay 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --do_eval true \
    --metric_for_best_model "eval_loss" \
    --load_best_model_at_end true 