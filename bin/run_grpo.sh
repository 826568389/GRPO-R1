#!/bin/bash
set -e  # 如果任何命令失败则退出

# 检查参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_size>"
    echo "Available model sizes: 0.5b, 1.5b, 3b, 7b"
    exit 1
fi

MODEL_SIZE=$1
case ${MODEL_SIZE,,} in  # 转换为小写
    "0.5b"|"0dot5b")
        CONFIG_NAME="0dot5B"
        ;;
    "1.5b"|"1dot5b")
        CONFIG_NAME="1dot5B"
        ;;
    "3b")
        CONFIG_NAME="3B"
        ;;
    "7b")
        CONFIG_NAME="7B"
        ;;
    *)
        echo "Error: Invalid model size. Available options: 0.5b, 1.5b, 3b, 7b"
        exit 1
        ;;
esac

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

#GPU并行数0,1,2,3
GPU_NUM=3

# 运行训练
echo "Starting training for ${MODEL_SIZE} model..."
echo "tail -f ${OUTPUT_DIR}/grpo_r1_${CONFIG_NAME}_sampling.log 查看训练日志" 
accelerate launch \
    --config_file "${PROJECT_ROOT}/config/zero3.json" \
    --num_processes="${GPU_NUM}" "${PROJECT_ROOT}/src/grpo_r1/grpo.py" \
    --config "${PROJECT_ROOT}/config/GRPO_R1_zero_${CONFIG_NAME}_config.yaml" \
    > "${OUTPUT_DIR}/grpo_r1_${CONFIG_NAME}_sampling.log" 2>&1 

echo "Training completed. Log file: ${OUTPUT_DIR}/grpo_r1_${CONFIG_NAME}_sampling.log" 
