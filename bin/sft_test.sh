#!/bin/bash

# 获取项目根目录
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# 添加项目根目录到PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 禁用各种警告信息
export TRANSFORMERS_VERBOSITY=error
export TORCH_SHOW_WARNING=0
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"

MODEL_PATH="output/Qwen2.5-0.5B-Instruct-sft-lora/final_model"

# 执行测试命令
python ${PROJECT_ROOT}/src/sft/test.py \
    --model_path ${MODEL_PATH} \
    --fp16 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 512 \
    --repetition_penalty 1.1 \
    --system_prompt "你是一个有用的AI助手。请简洁、准确地回答问题。避免重复和冗长的回答。" 