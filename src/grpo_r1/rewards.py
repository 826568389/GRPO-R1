"""
GRPO训练中使用的奖励函数模块
包含了用于评估模型生成质量的各种奖励函数
"""

import re
from typing import Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(completions, solution, **kwargs):
    """
    检查生成的答案是否与标准答案匹配的奖励函数
    
    参数:
        completions: 模型生成的完成序列列表
        solution: 标准答案列表
        **kwargs: 其他参数
        
    返回:
        rewards: 每个完成序列的奖励值列表，匹配为1.0，不匹配为0.0
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # 解析标准答案的LaTeX表达式
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # 解析生成答案的LaTeX表达式，要求格式正确
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # 验证答案是否匹配
            reward = float(verify(answer_parsed, gold_parsed))
            print('-'*100)
            print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        else:
            # 如果无法解析标准答案，默认给出满分
            reward = 1.0
            print("无法解析标准答案: ", sol)
        rewards.append(reward)

    print('\n准确率奖励:', rewards)

    return rewards


def format_reward(completions, **kwargs):
    """
    检查生成的答案是否符合指定格式的奖励函数
    要求答案包含<think>和<answer>标签
    
    参数:
        completions: 模型生成的完成序列列表
        **kwargs: 其他参数
        
    返回:
        rewards: 每个完成序列的奖励值列表，格式正确为1.0，错误为0.0
    """
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    print('-'*100)
    print('\n格式奖励:', rewards)
    return rewards


def reasoning_steps_reward(completions, **kwargs):
    """
    检查答案是否包含清晰的步骤性推理的奖励函数
    
    识别以下格式:
    - "Step 1:", "Step 2:" 等步骤标记
    - "1.", "2." 等数字列表
    - "-" 或 "*" 开头的项目符号
    - "First,", "Second,", "Next,", "Finally," 等过渡词
    
    参数:
        completions: 模型生成的完成序列列表
        **kwargs: 其他参数
        
    返回:
        rewards: 每个完成序列的奖励值列表，根据识别到的步骤数量给出0.0-1.0的奖励
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # 使用魔法数字3作为标准步骤数，超过3步给满分，否则按比例给分
    return [min(1.0, count / 3) for count in matches]

def length_reward(completions: list[Dict[str, str]], solutions: list[str], **kwargs) -> float:
    """
    根据答案正确度和长短进行奖励的奖励函数
    
    按以下公式计算奖励:
    - 正确答案：reward = 0.5 - (len - min_len)/(max_len - min_len)
    - 错误答案：reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    
    参数:
        completions: 模型生成的完成序列列表
        **kwargs: 其他参数
        
    返回:
        rewards: 每个完成序列的奖励值列表，值在0-0.5之间
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards

# 奖励函数注册表
REWARD_FUNCS_REGISTRY = {
    "accuracy": accuracy_reward,      # 答案准确性奖励
    "format": format_reward,          # 格式规范性奖励
    "reasoning_steps": reasoning_steps_reward,  # 推理步骤完整性奖励
    "length_reward": length_reward,      # 答案长度奖励
}
