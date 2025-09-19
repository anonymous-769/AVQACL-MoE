import re
import json
import os
import logging
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_log_data(log_content):
    # 更灵活的文件路径匹配模式
    data_path_pattern = r'开始评估，数据路径:\s*(.*?\.json)'
    batch_pattern = r'处理批次 (\d+)/\d+'
    avg_sim_pattern = r'平均相似度:\s*\[(.*?)\]'
    modal_sim_pattern = r'  (\w+):\s*\[(.*?)\]'
    expert_pattern = r'最终选择的专家:\s*(\d+)'
    result_id_pattern = r"结果:\s*\{.*?'id':\s*'(\d+)'"

    # 初始化数据结构
    results = {}
    current_file = None
    current_batch = None
    file_count = 0
    batch_count = 0

    # 逐行处理日志内容
    for line_num, line in enumerate(log_content.split('\n'), 1):
        # 提取数据路径 - 更灵活的匹配
        data_path_match = re.search(data_path_pattern, line)
        if data_path_match:
            full_path = data_path_match.group(1).strip()
            current_file = os.path.basename(full_path)
            if current_file not in results:
                results[current_file] = []
                file_count += 1
                logger.info(f"找到新文件: {current_file} (行号: {line_num})")
            continue
        
        # 提取批次号
        batch_match = re.search(batch_pattern, line)
        if batch_match and current_file:
            batch_index = int(batch_match.group(1))
            current_batch = {
                "batch_index": batch_index,
                "average_similarity": None,
                "modal_similarities": {},
                "selected_expert": None,
                "result_id": None
            }
            results[current_file].append(current_batch)
            batch_count += 1
            logger.debug(f"找到批次: {batch_index} (行号: {line_num})")
            continue
        
        # 提取平均相似度
        if current_batch and current_batch["average_similarity"] is None:
            avg_sim_match = re.search(avg_sim_pattern, line)
            if avg_sim_match:
                sim_str = avg_sim_match.group(1).replace("'", "").replace('"', "")
                try:
                    current_batch["average_similarity"] = [
                        float(x.strip()) for x in sim_str.split(',')
                    ]
                    logger.debug(f"提取平均相似度: {current_batch['average_similarity']} (行号: {line_num})")
                except ValueError:
                    logger.warning(f"无法解析平均相似度: {sim_str} (行号: {line_num})")
                continue
        
        # 提取模态相似度
        modal_sim_match = re.search(modal_sim_pattern, line)
        if modal_sim_match and current_batch:
            modal = modal_sim_match.group(1).strip()
            values_str = modal_sim_match.group(2).replace("'", "").replace('"', "")
            try:
                values = [float(x.strip()) for x in values_str.split(',')]
                current_batch["modal_similarities"][modal] = values
                logger.debug(f"提取{modal}相似度: {values} (行号: {line_num})")
            except ValueError:
                logger.warning(f"无法解析{modal}相似度: {values_str} (行号: {line_num})")
            continue
        
        # 提取专家选择
        expert_match = re.search(expert_pattern, line)
        if expert_match and current_batch:
            current_batch["selected_expert"] = int(expert_match.group(1))
            logger.debug(f"提取专家选择: {current_batch['selected_expert']} (行号: {line_num})")
            continue
        
        # 提取结果ID - 更灵活的匹配
        result_id_match = re.search(result_id_pattern, line)
        if result_id_match and current_batch:
            current_batch["result_id"] = result_id_match.group(1)
            logger.debug(f"提取结果ID: {current_batch['result_id']} (行号: {line_num})")
            continue
    
    logger.info(f"处理完成: 找到 {file_count} 个文件, {batch_count} 个批次")
    return results

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_files = 0
    
    for filename, data in results.items():
        # 确保文件名以.json结尾
        if not filename.endswith('.json'):
            filename += '.json'
        
        output_path = os.path.join(output_dir, "anchor_select_" + filename)
        
        # 只保存有数据的文件
        if data:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            saved_files += 1
            logger.info(f"保存文件: {output_path} ({len(data)} 个批次)")
        else:
            logger.warning(f"跳过空文件: {filename}")
    
    return saved_files

def parse_arguments():
    parser = argparse.ArgumentParser(description='评估日志数据提取工具')
    parser.add_argument('--log_files', nargs='+', required=True,
                        help='日志文件路径列表，支持多个文件')
    parser.add_argument('--output_dir', required=True,
                        help='输出目录路径')
    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')
    return parser.parse_args()

# 主处理流程
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 读取日志文件
    log_content = ""
    successful_files = 0
    
    for log_file_path in args.log_files:
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_content += f.read() + "\n"
            logger.info(f"成功读取日志文件: {log_file_path}")
            successful_files += 1
        except Exception as e:
            logger.error(f"读取日志文件失败: {log_file_path} - {e}")
            # 如果某个文件读取失败，不退出，继续处理其他文件
            continue

    if not log_content:
        logger.error("所有日志文件读取失败或内容为空，无法进行评估。")
        exit(1)
    
    logger.info(f"成功读取 {successful_files}/{len(args.log_files)} 个日志文件")
    
    # 提取数据
    logger.info("开始提取日志数据...")
    results = extract_log_data(log_content)
    
    # 打印提取的文件信息
    for filename, batches in results.items():
        logger.info(f"文件: {filename} - 提取到 {len(batches)} 个批次")
    
    # 保存结果
    saved_files = save_results(results, args.output_dir)
    
    # 打印统计信息
    total_batches = sum(len(batches) for batches in results.values())
    logger.info(f"处理完成! 共处理 {total_batches} 个批次, 保存 {saved_files} 个文件")
    
    # 检查空文件
    empty_files = [f for f, d in results.items() if not d]
    if empty_files:
        logger.warning(f"以下文件没有提取到数据: {', '.join(empty_files)}")
    
    logger.info(f"结果已保存到 {args.output_dir} 目录")