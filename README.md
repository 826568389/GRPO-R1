# GRPO-R1

GRPO-R1 是一个综合性的大语言模型训练框架，支持两种主要训练模式：
1. 基于 GRPO (Generative Reward-Penalty Optimization) 算法的强化学习训练，通过奖惩机制优化模型的生成质量
2. 基于 DeepSpeed ZeRO-3 的监督微调 (SFT) 训练，支持全量参数微调、LoRA 和 QLoRA 等多种微调方式

该框架针对不同规模的语言模型提供了优化的训练方案，可以满足从小型到大型模型的各种训练需求。

## 特性

- 🚀 支持多种规模的模型训练 (0.5B, 1.5B, 3B, 7B)
- 💫 基于 DeepSpeed ZeRO-3 的高效分布式训练
- 🔄 支持 VLLM 加速推理
- 📊 集成 Wandb 可视化训练过程
- 🛠 完整的训练评估流程
- 🆕 支持 SFT (Supervised Fine-Tuning) 训练
  - 🔧 支持全量参数微调和 LoRA/QLoRA 训练
  - 📈 自动记录训练指标和模型参数统计
  - ⚡ 集成 DeepSpeed ZeRO-3 优化
  - 🎯 支持训练过程可视化和断点续训

## 新增功能

### SFT 训练增强
- ✨ 新增全参数微调训练脚本 `bin/run_sft.sh`
- 🚀 新增 LoRA 训练支持 `bin/run_sft_lora.sh`
- 📊 优化训练监控和日志记录
- 💾 支持训练过程安全中断和恢复

### 模型测试功能
新增模型测试脚本 `bin/sft_test.sh`，支持以下特性：
- 🔄 支持加载全参数微调和 LoRA 模型
- 📝 流式输出生成结果
- ⚙️ 丰富的生成参数控制：
  - Temperature（温度）控制
  - Top-p 采样
  - 重复惩罚机制
  - 最大生成长度限制
- 💬 支持自定义系统提示语

### 使用示例

1. 全参数微调训练：
```bash
bash bin/run_sft.sh
```

2. LoRA 训练：
```bash
bash bin/run_sft_lora.sh
```

3. 模型测试：
```bash
bash bin/sft_test.sh
```

测试脚本支持的参数：
```bash
python src/sft/test.py \
    --model_path <模型路径> \
    --fp16 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 512 \
    --repetition_penalty 1.1 \
    --system_prompt "自定义系统提示语"
```

## 环境要求

- Python >= 3.10.9
- CUDA >= 11.8
- 显存要求:
  - 0.5B 模型: 至少 2 x 24GB GPU
  - 1.5B 模型: 至少 2 x 24GB GPU
  - 3B 模型: 至少 4 x 40GB GPU
  - 7B 模型: 至少 8 x 80GB GPU

## 快速开始

1. 克隆仓库:
```bash
git clone https://github.com/826568389/GRPO-R1
cd GRPO-R1
```

2. 创建并激活环境:
```bash
conda create -n grpo python=3.11
conda activate grpo
```

3. 安装依赖:
```bash
pip install -r requirements.txt
pip install -e ".[dev]"  # 开发模式安装
```

4. 运行训练:

### GRPO 训练
```bash
# 使用统一的训练脚本，指定模型大小
bash bin/run_grpo.sh 0.5b  # 训练 0.5B 模型
bash bin/run_grpo.sh 1.5b  # 训练 1.5B 模型
bash bin/run_grpo.sh 3b    # 训练 3B 模型
bash bin/run_grpo.sh 7b    # 训练 7B 模型
```

GRPO 训练参数说明：
- 模型规格：支持 0.5b/1.5b/3b/7b 四种规格
- 训练数据：默认使用 X-R1-750 数据集
- 输出目录：默认保存在 `output/grpo_model_{size}`

### SFT 训练

1. 全参数微调：
```bash
# 修改 run_sft.sh 中的配置
vim bin/run_sft.sh

# 主要配置项：
# - model_name_or_path: 预训练模型路径
# - dataset_name: 训练数据集路径
# - output_dir: 输出目录
# - batch_size: 训练批次大小
# - learning_rate: 学习率

# 运行训练
bash bin/run_sft.sh
```

2. LoRA 训练：
```bash
# 修改 run_sft_lora.sh 中的配置
vim bin/run_sft_lora.sh

# 主要配置项：
# - model_name_or_path: 基础模型路径
# - dataset_name: 训练数据集路径
# - output_dir: 输出目录
# - lora_r: LoRA 秩
# - lora_alpha: LoRA alpha
# - lora_dropout: LoRA dropout
# - lora_target_modules: 目标模块

# 运行训练
bash bin/run_sft_lora.sh
```

3. 测试训练效果：
```bash
# 修改 sft_test.sh 中的模型路径
vim bin/sft_test.sh

# 运行测试
bash bin/sft_test.sh

# 或者直接指定参数运行
python src/sft/test.py \
    --model_path output/Qwen2.5-0.5B-Instruct-sft-lora/final_model \
    --fp16 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 512 \
    --repetition_penalty 1.1 \
    --system_prompt "你是一个有用的AI助手。请简洁、准确地回答问题。"
```

### 训练监控

1. 查看训练日志：
```bash
tail -f output/*/training.log
```

2. 监控训练进度：
- 访问 Wandb 页面查看实时训练指标
- 查看模型保存目录下的 checkpoint 文件
- 通过日志观察 loss 变化和训练速度

3. 中断和恢复：
- 训练过程支持安全中断（Ctrl+C）
- 使用 `--resume_from_checkpoint` 参数恢复训练

## 数据集下载地址

# sft 训练
https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/tree/main
# grpo 训练
https://huggingface.co/datasets/xiaodongguaAIGC/X-R1-750

## 项目结构

```
.
├── bin/                    # 脚本目录
│   ├── run_grpo.sh        # GRPO 训练脚本
│   ├── run_sft.sh         # 全参数微调训练脚本
│   ├── run_sft_lora.sh    # LoRA 训练脚本
│   └── sft_test.sh        # 模型测试脚本
├── config/                 # 配置文件目录
│   ├── sft_zero3.json     # SFT DeepSpeed ZeRO-3 配置
│   ├── zero1.yaml         # GRPO ZeRO-1 配置
│   ├── zero2.yaml         # GRPO ZeRO-2 配置
│   ├── zero3.yaml         # GRPO ZeRO-3 配置
│   └── GRPO_R1_*.yaml     # GRPO 模型训练配置
├── src/                    # 源代码目录
│   ├── grpo_r1/           # GRPO 核心实现
│   └── sft/               # SFT 相关代码
│       ├── configs.py     # SFT 配置类定义
│       ├── train.py       # SFT 训练入口
│       ├── trainer.py     # SFT 训练器实现
│       └── test.py        # SFT 测试脚本
└── output/                # 模型输出目录
    ├── Qwen2.5-0.5B-Instruct-sft-full/    # 全参数微调输出
    └── Qwen2.5-0.5B-Instruct-sft-lora/    # LoRA 训练输出
```

主要目录说明：
- `bin/`: 包含所有训练和测试脚本
- `config/`: 包含 DeepSpeed 和模型训练配置
- `src/`: 源代码目录
  - `grpo_r1/`: GRPO 算法实现
  - `sft/`: SFT 训练相关实现
- `output/`: 训练结果输出目录

## 主要依赖

- accelerate >= 1.2.1
- torch == 2.5.1
- transformers
- deepspeed == 0.15.4
- vllm == 0.7.1
- trl == 0.14.0
- wandb >= 0.19.1

## 训练配置说明

- **ZeRO 配置**: 提供了三种 ZeRO 配置(1/2/3)，可根据硬件资源选择
- **模型配置**: 针对不同规模模型提供了优化的配置文件
- **分布式训练**: 支持多 GPU 训练，自动适配显存
- **SFT 训练特性**:
  - 支持全量参数微调和 LoRA/QLoRA 训练模式
  - 自动记录训练损失、学习率、梯度等指标
  - 支持模型参数量统计和训练进度可视化
  - 提供断点续训和模型检查点保存功能
  - 集成 DeepSpeed ZeRO-3 优化，提升训练效率

## 许可证

本项目采用 Apache License 2.0 许可证。详情请参见 LICENSE 文件。

## 贡献者

- 826568389 (826568389@qq.com)
- 2662007798 (2662007798@qq.com)
- supertpx

## 问题反馈

如果您在使用过程中遇到任何问题，欢迎通过以下方式反馈：

1. 在 GitHub Issues 中提交问题
2. 发送邮件至维护者邮箱
3. 提交 Pull Request

## 致谢

感谢所有为本项目做出贡献的开发者。