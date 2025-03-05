"""
SFT的配置模块
定义了训练过程中需要的各种配置类
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union

from transformers import TrainingArguments


@dataclass
class DeepSpeedConfig:
    """DeepSpeed ZeRO 配置类"""
    zero_stage: int = field(
        default=3,
        metadata={"help": "DeepSpeed ZeRO stage (0-3)"}
    )
    offload_optimizer: bool = field(
        default=True,
        metadata={"help": "是否将优化器状态卸载到CPU"}
    )
    offload_param: bool = field(
        default=True,
        metadata={"help": "是否将参数卸载到CPU"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "梯度累积步数"}
    )
    gradient_clipping: float = field(
        default=1.0,
        metadata={"help": "梯度裁剪阈值"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "是否使用FP16训练"}
    )


@dataclass
class SFTArguments(TrainingArguments):
    """SFT训练参数配置类"""
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "预训练模型的路径或huggingface hub ID"}
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "训练数据集名称"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "学习率"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "训练轮数"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "每个设备的训练批次大小"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "梯度累积步数"}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "模型保存策略"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "每多少步保存一次模型"}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "每多少步记录一次日志"}
    )
    deepspeed_config: DeepSpeedConfig = field(
        default_factory=DeepSpeedConfig,
        metadata={"help": "DeepSpeed配置"}
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "wandb运行名称，如果不指定则自动生成"}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "预热步数占总步数的比例"}
    )
    training_mode: str = field(
        default="full",
        metadata={"help": "训练模式: 'full' (全参数微调), 'lora' (LoRA), 'qlora' (QLoRA)"}
    )
    preprocessing_batch_size: int = field(
        default=16,
        metadata={"help": "数据预处理时的批处理大小"}
    )
    # 特殊标记参数
    special_tokens: str = field(
        default="",
        metadata={"help": "要添加到词表的特殊标记，用逗号分隔"}
    )
    # LoRA 参数
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA秩"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha参数"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout概率"}
    )
    lora_target_modules: str = field(
        default="c_attn,c_proj",
        metadata={"help": "要应用LoRA的模块名称，用逗号分隔。Qwen模型使用: c_attn,c_proj"}
    )
    quantization_bits: Optional[int] = field(
        default=None,
        metadata={"help": "量化位数，用于QLoRA，可选值：4或8"}
    )
    use_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "是否使用梯度检查点以节省显存"}
    )
    # 生成相关参数
    predict_with_generate: bool = field(
        default=False,
        metadata={"help": "是否使用生成模式进行预测"}
    )
    generation_max_length: int = field(
        default=128,
        metadata={"help": "生成时的最大长度"}
    )
    generation_num_beams: int = field(
        default=4,
        metadata={"help": "生成时使用的beam search数量"}
    )
    # 添加安全序列化相关参数
    safe_serialization: bool = field(
        default=True,
        metadata={"help": "是否使用安全的序列化方式保存模型"}
    )
    weights_only: bool = field(
        default=True,
        metadata={"help": "加载检查点时是否只加载权重"}
    )
    torch_compile: bool = field(
        default=False,
        metadata={"help": "是否使用torch.compile加速"}
    )
    
    @property
    def lora_target_modules_list(self) -> List[str]:
        """将逗号分隔的字符串转换为列表"""
        return [m.strip() for m in self.lora_target_modules.split(",") if m.strip()]
        
    @property
    def special_tokens_list(self) -> List[str]:
        """将逗号分隔的特殊标记字符串转换为列表"""
        return [token.strip() for token in self.special_tokens.split(",") if token.strip()] 