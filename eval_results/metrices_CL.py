import json
import os
import numpy as np

def calculate_accuracy_from_json(json_path):
    """计算单个JSON文件的准确率"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    correct_count = 0
    total_count = 0
    
    for sample_id, sample_data in data.items():
        # 每个样本的第一个字典包含预测评价
        if isinstance(sample_data, list) and len(sample_data) > 0:
            pred_info = sample_data[0]
            if pred_info.get('pred') == 'yes':
                correct_count += 1
            total_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0

def build_accuracy_matrix(results_dir, total_tasks=4):
    """
    构建准确率矩阵
    :param results_dir: 结果根目录
    :param total_tasks: 总任务数
    :return: 准确率矩阵 (stage_index, task_index)
    """
    # 初始化准确率矩阵 (行：训练阶段，列：任务ID)
    accuracy_matrix = np.full((total_tasks, total_tasks), np.nan)
    
    # 遍历每个训练阶段 (1_experts, 2_experts, ...)
    for stage_idx in range(1, total_tasks + 1):
        stage_dir = os.path.join(results_dir, f"{stage_idx}_experts")
        
        if not os.path.exists(stage_dir):
            continue
            
        # 遍历当前阶段的所有任务结果
        for task_idx in range(0, stage_idx):  # 阶段k包含任务0到k-1
            json_file = os.path.join(stage_dir, f"eval_metrices_task{task_idx}.json")
            
            if os.path.exists(json_file):
                accuracy = calculate_accuracy_from_json(json_file)
                # 阶段索引从0开始：阶段1 -> 索引0，阶段2 -> 索引1...
                # 任务索引：任务0 -> 索引0，任务1 -> 索引1...
                accuracy_matrix[stage_idx-1, task_idx] = accuracy
    
    return accuracy_matrix

def calculate_ma(accuracy_matrix):
    """
    计算平均准确率(MA)
    :param accuracy_matrix: 准确率矩阵
    :return: MA值
    """
    # 最终阶段的所有任务准确率 (最后一行)
    final_accuracies = accuracy_matrix[-1]
    
    # 过滤NaN值 (未测试的任务)
    valid_accuracies = final_accuracies[~np.isnan(final_accuracies)]
    
    # MA = 1/T * Σ(最终阶段各任务准确率)
    return np.mean(valid_accuracies) if len(valid_accuracies) > 0 else 0.0

def calculate_af(accuracy_matrix):
    """
    计算平均遗忘率(AF)
    :param accuracy_matrix: 准确率矩阵
    :return: AF值
    """
    total_stages = accuracy_matrix.shape[0]
    total_forgetting = 0.0
    valid_stages = 0
    
    # 遍历每个阶段 (从阶段2开始)
    for stage_idx in range(1, total_stages):
        # 当前阶段索引 (阶段t)
        current_stage = stage_idx  # 对应矩阵行索引
        
        # 历史任务数 (0到t-1)
        historical_tasks = stage_idx
        stage_forgetting = 0.0
        valid_tasks = 0
        
        # 遍历所有历史任务
        for task_idx in range(0, historical_tasks):
            # 获取历史最高准确率 (阶段1到t-1)
            historical_accuracies = accuracy_matrix[:stage_idx, task_idx]
            
            # 过滤NaN值
            valid_historical = historical_accuracies[~np.isnan(historical_accuracies)]
            
            if len(valid_historical) > 0:
                max_historical = np.max(valid_historical)
                current_accuracy = accuracy_matrix[stage_idx, task_idx]
                
                # 计算遗忘量 = 历史最高 - 当前准确率
                if not np.isnan(current_accuracy):
                    stage_forgetting += (max_historical - current_accuracy)
                    valid_tasks += 1
        
        # 当前阶段的平均遗忘率
        if valid_tasks > 0:
            stage_forgetting /= valid_tasks
            total_forgetting += stage_forgetting
            valid_stages += 1
    
    # AF = 1/(T-1) * Σ(各阶段平均遗忘率)
    return total_forgetting / valid_stages if valid_stages > 0 else 0.0

import argparse

# ====================== 主执行流程 ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算连续学习指标 (MA, AF) 的工具')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='结果根目录，包含不同专家阶段的评估结果')
    parser.add_argument('--total_tasks', type=int, default=5,
                        help='总任务数，默认为5')
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir
    TOTAL_TASKS = args.total_tasks
    
    # 步骤1: 构建准确率矩阵
    accuracy_matrix = build_accuracy_matrix(RESULTS_DIR, TOTAL_TASKS)
    print("准确率矩阵:")
    print(accuracy_matrix)
    
    # 步骤2: 计算MA
    ma_value = calculate_ma(accuracy_matrix)
    print(f"\nMA (平均准确率) = {ma_value:.4f}")
    
    # 步骤3: 计算AF
    af_value = calculate_af(accuracy_matrix)
    print(f"AF (平均遗忘率) = {af_value:.4f}")

    # 步骤4: 打印最终指标
    print("\n最终指标:")
    print(f"MA: {ma_value*100:.2f}%")
    print(f"AF: {af_value*100:.2f}%")