#!/bin/bash
set -e  # 如果任何命令失败则退出

export ACCELERATE_LOG_LEVEL=info

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

accelerate launch \
    --config_file "${PROJECT_ROOT}/config/GRPO3.yaml" \
    --num_processes=3 "${PROJECT_ROOT}/src/grpo_r1/grpo.py" \
    --config "${PROJECT_ROOT}/config/GRPO_R1_zero_0dot5B_config.yaml" \
    > "${OUTPUT_DIR}/grpo_r1_0dot5B_sampling.log" 2>&1  # 同时捕获标准错误输出 
