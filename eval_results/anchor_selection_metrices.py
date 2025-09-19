import json
import os
import glob
import re
import numpy as np
import pandas as pd
import argparse

# 定义文件路径
anchor_dir = './AVQA_CL/Case_study'
output_dir = './AVQA_CL/Analysis_Results'

def parse_arguments():
    parser = argparse.ArgumentParser(description='提取预测结果到日志文件工具')
    parser.add_argument('--anchor_dir', type=str, required=True,
                        help='指定anchor文件所在的目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_arguments()
    anchor_dir = args.anchor_dir
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    # 获取所有anchor文件
    anchor_files = glob.glob(os.path.join(anchor_dir, 'anchor_select_test_*.json'))
    print(f"找到 {len(anchor_files)} 个anchor文件")

    # 第一步：确定专家数量
    # 收集所有任务类型ID和专家ID
    task_ids = set()
    expert_ids = set()

    for anchor_file in anchor_files:
        # 从文件名中提取任务类型ID
        match = re.search(r'_(\d+)\.json$', anchor_file)
        if match:
            task_id = int(match.group(1))
            task_ids.add(task_id)
        
        # 加载文件数据收集专家ID
        try:
            with open(anchor_file, 'r') as f:
                anchor_data = json.load(f)
            for item in anchor_data:
                if 'selected_expert' in item:
                    expert_ids.add(item["selected_expert"])
        except:
            continue

    # 确定专家数量
    all_ids = task_ids.union(expert_ids)
    if all_ids:
        num_experts = max(all_ids) + 1
    else:
        num_experts = 5  # 默认值

    print(f"确定专家数量为: {num_experts}")

    # 初始化全局统计
    global_stats = {
        'match_correct': 0,
        'match_total': 0,
        'mismatch_correct': 0,
        'mismatch_total': 0,
        'task_expert_matrix': np.zeros((num_experts, num_experts)),  # 动态大小的矩阵
        'task_expert_counts': np.zeros((num_experts, num_experts))   # 计数矩阵
    }

    # 处理每个文件
    for anchor_file in anchor_files:
        print(f"\n处理文件: {os.path.basename(anchor_file)}")
        
        # 从文件名中提取任务类型ID
        match = re.search(r'_(\d+)\.json$', anchor_file)
        if not match:
            print(f"无法从文件名解析任务类型ID: {anchor_file}")
            continue
            
        task_id = int(match.group(1))
        print(f"文件对应的任务类型ID: {task_id}")
        
        # 加载文件数据
        try:
            with open(anchor_file, 'r') as f:
                anchor_data = json.load(f)
            print(f"成功加载数据，包含 {len(anchor_data)} 个条目")
        except Exception as e:
            print(f"加载文件失败: {str(e)}")
            continue
        
        # 初始化文件级统计
        file_stats = {
            'match_correct': 0,
            'match_total': 0,
            'mismatch_correct': 0,
            'mismatch_total': 0,
            'task_id': task_id
        }
        
        # 处理每个条目
        for item in anchor_data:
            if 'prediction' not in item:
                continue
                
            result_id = item["result_id"]
            selected_expert = item["selected_expert"]
            prediction = item["prediction"]
            
            # 计算正确性 (prediction为"yes"表示正确)
            correct = 1 if prediction == "yes" else 0
            
            # 确保ID在矩阵范围内
            if task_id < num_experts and selected_expert < num_experts:
                # 更新矩阵统计
                global_stats['task_expert_matrix'][task_id][selected_expert] += correct
                global_stats['task_expert_counts'][task_id][selected_expert] += 1
                
                # 判断是否匹配
                if selected_expert == task_id:
                    file_stats['match_total'] += 1
                    file_stats['match_correct'] += correct
                    global_stats['match_total'] += 1
                    global_stats['match_correct'] += correct
                else:
                    file_stats['mismatch_total'] += 1
                    file_stats['mismatch_correct'] += correct
                    global_stats['mismatch_total'] += 1
                    global_stats['mismatch_correct'] += correct
            else:
                print(f"警告: 任务ID {task_id} 或专家ID {selected_expert} 超出范围 (0-{num_experts-1})")
        
        # 计算文件级准确率
        if file_stats['match_total'] > 0:
            match_accuracy = file_stats['match_correct'] / file_stats['match_total']
        else:
            match_accuracy = 0.0
            
        if file_stats['mismatch_total'] > 0:
            mismatch_accuracy = file_stats['mismatch_correct'] / file_stats['mismatch_total']
        else:
            mismatch_accuracy = 0.0
        
        print(f"任务类型 {task_id} 匹配准确率: {match_accuracy:.4f} ({file_stats['match_correct']}/{file_stats['match_total']})")
        print(f"任务类型 {task_id} 不匹配准确率: {mismatch_accuracy:.4f} ({file_stats['mismatch_correct']}/{file_stats['mismatch_total']})")
        
        # 保存文件级结果
        file_stats['match_accuracy'] = match_accuracy
        file_stats['mismatch_accuracy'] = mismatch_accuracy
        file_stats['accuracy_difference'] = match_accuracy - mismatch_accuracy
        
        # 保存到文件
        result_file = os.path.join(output_dir, f"analysis_{os.path.basename(anchor_file)}")
        with open(result_file, 'w') as f:
            json.dump(file_stats, f, indent=2)
        print(f"已保存分析结果到: {result_file}")

    # 计算全局准确率
    if global_stats['match_total'] > 0:
        global_match_accuracy = global_stats['match_correct'] / global_stats['match_total']
    else:
        global_match_accuracy = 0.0

    if global_stats['mismatch_total'] > 0:
        global_mismatch_accuracy = global_stats['mismatch_correct'] / global_stats['mismatch_total']
    else:
        global_mismatch_accuracy = 0.0

    # 计算准确率矩阵
    accuracy_matrix = np.zeros((num_experts, num_experts))
    for i in range(num_experts):
        for j in range(num_experts):
            if global_stats['task_expert_counts'][i][j] > 0:
                accuracy_matrix[i][j] = global_stats['task_expert_matrix'][i][j] / global_stats['task_expert_counts'][i][j]
            else:
                accuracy_matrix[i][j] = 0.0

    # 计算比例矩阵（任务分配给专家的比例）
    proportion_matrix = np.zeros((num_experts, num_experts))
    for i in range(num_experts):
        total_count = np.sum(global_stats['task_expert_counts'][i])
        if total_count > 0:
            for j in range(num_experts):
                proportion_matrix[i][j] = global_stats['task_expert_counts'][i][j] / total_count
        else:
            proportion_matrix[i][j] = 0.0

    # 创建DataFrame用于输出
    task_labels = [f"任务类型 {i}" for i in range(num_experts)]
    expert_labels = [f"专家 {j}" for j in range(num_experts)]

    # 创建组合矩阵，同时显示准确率和比例
    combined_matrix = np.empty((num_experts, num_experts), dtype=object)
    for i in range(num_experts):
        for j in range(num_experts):
            proportion = proportion_matrix[i][j]
            if proportion > 0:
                acc = accuracy_matrix[i][j]
                # 标记匹配情况（对角线）
                if i == j:
                    combined_matrix[i][j] = f"{acc:.4f}* ({proportion:.4f})"
                else:
                    combined_matrix[i][j] = f"{acc:.4f} ({proportion:.4f})"
            else:
                combined_matrix[i][j] = "N/A"

    df_combined = pd.DataFrame(combined_matrix, index=task_labels, columns=expert_labels)

    # 保存全局结果
    # 将NumPy数组转换为列表以便JSON序列化
    global_stats['task_expert_matrix'] = global_stats['task_expert_matrix'].tolist()
    global_stats['task_expert_counts'] = global_stats['task_expert_counts'].tolist()
    global_stats['accuracy_matrix'] = accuracy_matrix.tolist()
    global_stats['proportion_matrix'] = proportion_matrix.tolist()

    global_stats['global_match_accuracy'] = global_match_accuracy
    global_stats['global_mismatch_accuracy'] = global_mismatch_accuracy
    global_stats['accuracy_difference'] = global_match_accuracy - global_mismatch_accuracy

    global_result_file = os.path.join(output_dir, "global_analysis.json")
    with open(global_result_file, 'w') as f:
        json.dump(global_stats, f, indent=2)

    # 输出结果
    print("\n===== 全局分析结果 =====")
    print(f"匹配准确率: {global_match_accuracy:.4f} ({global_stats['match_correct']}/{global_stats['match_total']})")
    print(f"不匹配准确率: {global_mismatch_accuracy:.4f} ({global_stats['mismatch_correct']}/{global_stats['mismatch_total']})")
    print(f"准确率差异: {global_match_accuracy - global_mismatch_accuracy:.4f}")

    print("\n===== 准确率与分配比例矩阵 =====")
    print("行: 任务类型, 列: 专家, *表示匹配情况")
    print("格式: 准确率 (分配比例)")
    print(df_combined)

    # 保存矩阵到CSV
    matrix_file = os.path.join(output_dir, "match_accuracy_matrix.csv")
    df_combined.to_csv(matrix_file)
    print(f"\n已保存准确率与分配比例矩阵到: {matrix_file}")

    print("\n所有分析完成")