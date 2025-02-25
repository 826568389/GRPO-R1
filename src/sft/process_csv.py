"""
CSV和Excel数据处理脚本
用于将CSV或Excel格式的数据转换为JSONL格式
支持的列：input,output,instruction
"""

import csv
import json
import os
import logging
import argparse
import chardet
import pandas as pd
from typing import Dict, List, Any

# 设置日志格式
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def detect_file_type(file_path: str) -> str:
    """
    通过文件魔数检测文件类型
    """
    # 已知的文件魔数
    EXCEL_MAGIC = b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'  # xls
    XLSX_MAGIC = b'PK\x03\x04'  # xlsx (zip)
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
            
        if header.startswith(EXCEL_MAGIC):
            return 'xls'
        elif header.startswith(XLSX_MAGIC):
            return 'xlsx'
            
        # 尝试读取更多字节来检查UTF-8 BOM
        with open(file_path, 'rb') as f:
            content = f.read(4)
            if content.startswith(b'\xEF\xBB\xBF'):  # UTF-8 with BOM
                return 'csv-utf8-sig'
            
        return 'csv'  # 默认假设是CSV
        
    except Exception as e:
        logger.error(f"检测文件类型时出错: {str(e)}")
        return 'unknown'

def read_excel_or_csv(file_path: str) -> pd.DataFrame:
    """
    读取Excel或CSV文件
    """
    try:
        # 检测真实的文件类型
        file_type = detect_file_type(file_path)
        logger.info(f"检测到文件类型: {file_type}")
        
        if file_type in ['xls', 'xlsx']:
            try:
                logger.info(f"尝试使用Excel格式读取文件...")
                df = pd.read_excel(file_path)
                logger.info(f"成功读取Excel文件，列名: {list(df.columns)}")
                return df
            except Exception as e:
                logger.error(f"Excel读取失败: {str(e)}")
                raise
        
        # 如果不是Excel或读取失败，尝试CSV格式
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'gb2312', 'cp936']
        
        # 如果检测到UTF-8 with BOM，优先使用
        if file_type == 'csv-utf8-sig':
            encodings.insert(0, 'utf-8-sig')
        
        for encoding in encodings:
            try:
                logger.info(f"尝试使用 {encoding} 编码读取CSV文件...")
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"成功使用 {encoding} 读取CSV文件，列名: {list(df.columns)}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"使用 {encoding} 读取失败: {str(e)}")
                continue
                
        # 如果所有尝试都失败了
        raise ValueError(f"无法读取文件。文件类型: {file_type}，请确保文件格式正确。")
        
    except Exception as e:
        logger.error(f"读取文件失败: {str(e)}")
        raise

def validate_row(row: Dict[str, str], row_num: int) -> bool:
    """
    验证数据行的有效性
    """
    # 检查必要字段
    required_fields = ["input", "output", "instruction"]
    for field in required_fields:
        if field not in row or pd.isna(row[field]):
            logger.warning(f"第{row_num}行缺少必要字段或字段为空: {field}")
            return False
        if not str(row[field]).strip() or len(str(row[field]).strip()) < 2:
            logger.warning(f"第{row_num}行字段 {field} 为空或过短")
            return False
    return True

def process_file(input_file: str, output_file: str) -> None:
    """
    处理Excel或CSV文件并转换为JSONL格式
    
    Args:
        input_file: 输入文件路径（Excel或CSV）
        output_file: 输出JSONL文件路径
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 读取文件
        df = read_excel_or_csv(input_file)
        
        # 如果只有两列（input和output），自动添加instruction列
        if set(df.columns) == {"input", "output"}:
            logger.info("检测到只有input和output列，自动添加instruction列")
            df["instruction"] = "请根据问题提供准确的回答。"
        
        # 验证必要列是否存在
        required_fields = {"input", "output", "instruction"}
        if not required_fields.issubset(set(df.columns)):
            missing = required_fields - set(df.columns)
            raise ValueError(f"文件缺少必要列: {missing}")
        
        # 处理数据并写入JSONL
        valid_count = 0
        total_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for index, row in df.iterrows():
                total_count += 1
                row_num = index + 2  # 加2是因为Excel行号从1开始，还要算上标题行
                
                try:
                    # 转换为字典并清理数据
                    row_dict = row.to_dict()
                    cleaned_row = {
                        k: str(v).strip() if pd.notnull(v) else ""
                        for k, v in row_dict.items()
                    }
                    
                    # 验证数据有效性
                    if not validate_row(cleaned_row, row_num):
                        continue
                    
                    # 只保留必要字段
                    output_row = {
                        field: cleaned_row[field]
                        for field in required_fields
                    }
                    
                    # 写入JSONL格式
                    json.dump(output_row, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    valid_count += 1
                    
                except Exception as e:
                    logger.warning(f"处理第{row_num}行时出错: {str(e)}")
                    continue
                
                # 打印进度
                if total_count % 1000 == 0:
                    logger.info(f"已处理 {total_count} 行...")
        
        logger.info(f"处理完成！")
        logger.info(f"总行数: {total_count}")
        logger.info(f"有效行数: {valid_count}")
        logger.info(f"输出文件: {output_file}")
        
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        raise

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="将Excel或CSV格式数据转换为JSONL格式")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径（Excel或CSV）")
    parser.add_argument("--output", type=str, required=True, help="输出JSONL文件路径")
    
    args = parser.parse_args()
    
    # 处理文件
    process_file(args.input, args.output)

if __name__ == "__main__":
    main() 