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

# 模型配置
MODEL_TYPE="lora"  # 可选: "lora" 或 "base"
MERGE_LORA=true   # 是否合并LoRA权重到基础模型

# 基础模型路径
BASE_MODEL_PATH="/data/staryea/aigc_model/Qwen2.5-3B-Instruct"

if [ "$MODEL_TYPE" = "lora" ]; then
    # LoRA模型配置
    MODEL_PATH="/data/staryea/GRPO-R1/bin/output/Qwen2.5-3B-Instruct-Business-lora-030418/final_model"
    # 执行测试命令 (带基础模型路径)
    python ${PROJECT_ROOT}/src/sft/test.py \
        --model_path ${MODEL_PATH} \
        --base_model_path ${BASE_MODEL_PATH} \
        --bf16 \
        --merge_lora \
        --max_new_tokens 4096 \
        --temperature 0.7 \
        --top_p 0.9 \
        --repetition_penalty 1.1 \
        --system_prompt "你是一个专业的客服分类助手。请对用户的问题进行分类，并解释分类理由。"
else
    # 基础模型配置
    MODEL_PATH=${BASE_MODEL_PATH}
    # 执行测试命令 (不带基础模型路径)
    python ${PROJECT_ROOT}/src/sft/test.py \
        --model_path ${MODEL_PATH} \
        --bf16 \
        --temperature 0.7 \
        --top_p 0.9 \
        --max_new_tokens 4096 \
        --repetition_penalty 1.1 
fi
