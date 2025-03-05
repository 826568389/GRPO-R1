#!/bin/bash

# 获取项目根目录
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# 添加项目根目录到PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# wandb配置
export WANDB_PROJECT="model-evaluation"
export WANDB_WATCH="false"

# 模型配置
BASE_MODEL_PATH="/data/staryea/aigc_model/Qwen2.5-3B-Instruct"
MODEL_PATH="/data/staryea/GRPO-R1/bin/output/Qwen2.5-3B-Instruct-Business-lora-030419/final_model"
TEST_FILE="/data/staryea/GRPO-R1/bin/normalized_data/normalized_data_20250305_115703.jsonl"
OUTPUT_DIR="evaluation_results"

# 执行评估命令
python ${PROJECT_ROOT}/src/sft/evaluate.py \
    --model_path ${MODEL_PATH} \
    --base_model_path ${BASE_MODEL_PATH} \
    --test_file ${TEST_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --generate \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --num_beams 4 \
    --max_length 2048 \
    --wandb_project "model-evaluation" \
    --no_wandb \
    --wandb_name "eval-qwen-lora" 