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

## 环境要求

- Python >= 3.10.9
- CUDA >= 11.8
- 显存要求:
  - 0.5B 模型: 至少 2 x 24GB GPU
  - 1.5B 模型: 至少 4 x 24GB GPU
  - 3B 模型: 至少 4 x 24GB GPU
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
```bash
# 使用统一的训练脚本，指定模型大小
bash bin/run_grpo.sh 0.5b  # 训练 0.5B 模型
bash bin/run_grpo.sh 1.5b  # 训练 1.5B 模型
bash bin/run_grpo.sh 3b    # 训练 3B 模型
bash bin/run_grpo.sh 7b    # 训练 7B 模型

# SFT训练脚本
bash bin/run_sft.sh   # SFT训练 ，在脚本中修改 目录路径和数据集模型
```
## 数据集下载地址

# sft 训练
https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/tree/main
# grpo 训练
https://huggingface.co/datasets/xiaodongguaAIGC/X-R1-750

## 项目结构

```
.
├── bin/            # 训练脚本
├── config/         # 配置文件
│   ├── zero1.yaml     # ZeRO-1 配置
│   ├── zero2.yaml     # ZeRO-2 配置
│   ├── zero3.yaml     # ZeRO-3 配置
│   └── GRPO_R1_*.yaml # 模型训练配置
├── src/           # 源代码
│   └── grpo_r1/   # 核心实现
└── output/        # 输出目录
```

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