# GRPO-R1

GRPO-R1是一个基于深度学习的项目，专注于大规模语言模型的训练和优化。该项目使用了最新的AI技术和工具，包括Hugging Face的transformers库和各种优化框架。

## 项目特点

- 基于PyTorch框架
- 支持分布式训练（使用DeepSpeed）
- 集成了多个先进的AI训练和优化工具
- 支持高效的模型推理（使用VLLM）

## 系统要求

- Python >= 3.10.9
- CUDA 兼容的GPU（推荐）
- 足够的RAM和GPU内存

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/826568389/GRPO-R1
cd GRPO-R1
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 开发模式安装：
```bash
pip install -e ".[dev]"
```

## 主要依赖

- accelerate >= 1.2.1
- torch == 2.5.1
- transformers
- deepspeed == 0.15.4
- vllm == 0.7.1
- bitsandbytes >= 0.43.0
- huggingface-hub[cli] >= 0.19.2
- datasets >= 3.2.0

## 项目结构

```
.
├── config/         # 配置文件目录
├── scripts/        # 实用脚本
├── src/           # 源代码
│   └── grpo_r1/   # 核心代码
└── tests/         # 测试文件
```

## 许可证

本项目采用 Apache License 2.0 许可证。详情请参见 LICENSE 文件。

## 联系方式

- 作者：826568389 2662007798 supertpx
- 作者：826568389@qq.com 2662007798@qq.com
- 项目地址：https://github.com/826568389/GRPO-R1

## 贡献指南

欢迎提交问题和合并请求。在提交之前，请确保：

1. 更新测试用例
2. 更新文档
3. 遵循项目的代码风格
4. 所有测试都通过

## 致谢

感谢所有为本项目做出贡献的开发者。