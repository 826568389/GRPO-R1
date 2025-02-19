#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用的GPU
export MASTER_PORT=28500

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# 检查并创建输出目录（如果不存在）
OUTPUT_DIR="${PROJECT_ROOT}/output"
if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p "${OUTPUT_DIR}"
fi

# 添加项目根目录到PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# 设置 PyTorch 内存分配器配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 训练参数
MODEL_NAME_OR_PATH="/data/staryea/aigc_model/Qwen2.5-0.5B-Instruct"  # 替换为实际的模型路径
DATASET_NAME="/data/staryea/DeepSeek/dataset/BelleGroup/train_1k_CN"  # 替换为实际的数据集名称
OUTPUT_DIR="output/Qwen2.5-0.5B-Instruct-sft-full"  # 输出目录

# 训练模式配置
TRAINING_MODE="full"  # 可选: "full" (全参数微调), "lora" (LoRA), "qlora" (QLoRA)

# LoRA配置
LORA_R=8                # LoRA秩
LORA_ALPHA=16          # LoRA alpha参数
LORA_DROPOUT=0.05      # LoRA dropout概率

# QLoRA配置（仅在TRAINING_MODE="qlora"时使用）
QUANTIZATION_BITS=4    # 量化位数，可选4或8

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 设置基本的训练参数
TRAIN_ARGS="\
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --max_seq_length 2048 \
    --save_strategy steps \
    --save_steps 500 \
    --logging_steps 100 \
    --use_gradient_checkpointing true \
    --bf16 true \
    --training_mode $TRAINING_MODE"

# 根据训练模式设置不同的配置
if [ "$TRAINING_MODE" = "full" ]; then
    # 全参数微调使用 DeepSpeed
    export WANDB_PROJECT="sft-training-full"
    export WANDB_WATCH="gradients"
    export WANDB_LOG_MODEL="checkpoint"
    export WANDB_DISABLED="false"
    
    TRAIN_CMD="deepspeed --num_gpus=4 ${PROJECT_ROOT}/src/sft/train.py $TRAIN_ARGS \
    --deepspeed ${PROJECT_ROOT}/config/sft_zero3.json \
    --report_to wandb"
else
    # LoRA 和 QLoRA 不使用 DeepSpeed
    export WANDB_PROJECT="sft-training-${TRAINING_MODE}"
    export WANDB_WATCH="false"
    export WANDB_LOG_MODEL="false"
    export WANDB_DISABLED="false"
    
    # 添加 LoRA 相关参数
    TRAIN_ARGS="$TRAIN_ARGS \
    --lora_config.r $LORA_R \
    --lora_config.alpha $LORA_ALPHA \
    --lora_config.dropout $LORA_DROPOUT"
    
    # 如果是 QLoRA，添加量化参数
    if [ "$TRAINING_MODE" = "qlora" ]; then
        TRAIN_ARGS="$TRAIN_ARGS --quantization_bits $QUANTIZATION_BITS"
    fi
    
    TRAIN_CMD="torchrun --nproc_per_node=4 ${PROJECT_ROOT}/src/sft/train.py $TRAIN_ARGS \
    --report_to none"
fi

# 执行训练命令
eval $TRAIN_CMD 