import json
import os

def extract_1k_samples(input_file, output_file, num_samples=100):
    """
    从大的JSON文件中提取指定数量的样本并保存
    每行是一个独立的JSON对象
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出JSON文件路径
        num_samples (int): 要提取的样本数量
    """
    # 读取原始数据
    print(f"正在读取文件: {input_file}")
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"警告：第{i+1}行解析失败：{e}")
                continue
    
    print(f"提取了 {len(samples)} 条样本")
    
    # 保存到新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"已保存到文件: {output_file}")

if __name__ == "__main__":
    # 设置文件路径
    input_file = "/data/staryea/DeepSeek/dataset/CVE_QA/CVE_QA.json"
    output_file = "/data/staryea/DeepSeek/dataset/CVE_QA/CVE_QA_0.1k.json"
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 提取并保存数据
    extract_1k_samples(input_file, output_file)