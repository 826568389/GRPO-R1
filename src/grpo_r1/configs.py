# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GRPO-R1的配置模块
定义了训练过程中需要的各种配置类
"""

from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    GRPO算法的配置类
    继承自TRL库的GRPOConfig，添加了回调函数、基准测试等额外配置
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], 
        metadata={"help": "训练后要运行的基准测试列表"}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], 
        metadata={"help": "训练过程中要运行的回调函数列表"}
    )
    system_prompt: Optional[str] = field(
        default=None, 
        metadata={"help": "基准测试中使用的可选系统提示词"}
    )
    hub_model_revision: Optional[str] = field(
        default="main", 
        metadata={"help": "模型推送到Hub时使用的分支名称"}
    )
    overwrite_hub_revision: bool = field(
        default=False, 
        metadata={"help": "是否覆盖Hub上的现有版本"}
    )
    push_to_hub_revision: bool = field(
        default=False, 
        metadata={"help": "是否推送到Hub的特定分支/版本"}
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    监督微调(SFT)的配置类
    继承自TRL库的SFTConfig，添加了额外的配置选项
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], 
        metadata={"help": "训练后要运行的基准测试列表"}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], 
        metadata={"help": "训练过程中要运行的回调函数列表"}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "基准测试中使用的可选系统提示词"},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "模型推送到Hub时使用的分支名称"},
    )
    overwrite_hub_revision: bool = field(
        default=False, 
        metadata={"help": "是否覆盖Hub上的现有版本"}
    )
    push_to_hub_revision: bool = field(
        default=False, 
        metadata={"help": "是否推送到Hub的特定分支/版本"}
    )
