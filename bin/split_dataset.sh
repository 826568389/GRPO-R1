#!/bin/bash

# 获取项目根目录
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# 添加项目根目录到PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 数据路径配置
INPUT_FILE="/data/staryea/DeepSeek/dataset/BusinessData/sft_data/opt_sft_v4.jsonl"
OUTPUT_DIR="/data/staryea/DeepSeek/dataset/BusinessData/sft_data"
TEST_FILE_NAME="test.jsonl"
TEST_RATIO=0.1
SEED=42

# 执行数据集分割脚本
python ${PROJECT_ROOT}/src/sft/split_dataset.py \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --test_file_name ${TEST_FILE_NAME} \
    --test_ratio ${TEST_RATIO} \
    --seed ${SEED} 