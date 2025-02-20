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

# 执行测试命令
python ${PROJECT_ROOT}/src/sft/test.py \
    --model_path "output/Qwen2.5-0.5B-Instruct-sft-full/final_model" \
    --fp16 