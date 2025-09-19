import json
import os
import glob
import re
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='提取预测结果到日志文件工具')
    parser.add_argument('--anchor_dir', type=str, required=True,
                        help='指定anchor文件所在的目录')
    parser.add_argument('--eval_dir', type=str, required=True,
                        help='指定评估文件所在的目录')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    anchor_dir = args.anchor_dir
    eval_dir = args.eval_dir

    # 获取anchor文件
    anchor_files = glob.glob(os.path.join(anchor_dir, 'anchor_select_test_*.json'))
    print(f"找到的anchor文件: {anchor_files}")

    # 创建类别到任务ID的映射
    category_to_task = {}
    eval_files = os.listdir(eval_dir)
    print(f"评估目录中的文件: {eval_files}")

    for eval_file in eval_files:
        if not eval_file.startswith('task') or not eval_file.endswith('_results.json'):
            continue
        
        # 解析文件名格式: task{数字}_{类别}_results.json
        match = re.match(r'task(\d+)_([\w]+)_results\.json', eval_file)
        if match:
            task_id = match.group(1)
            category = match.group(2)
            category_to_task[category] = task_id
            print(f"映射: 类别 '{category}' -> 任务ID '{task_id}' (来自文件: {eval_file})")
        else:
            print(f"无法解析文件名: {eval_file}")

    print(f"建立的类别到任务ID映射: {category_to_task}")

    for anchor_file in anchor_files:
        print(f"\n处理文件: {anchor_file}")
        
        # 从文件名中提取类别
        match = re.search(r'anchor_select_test_([\w]+)\.json', os.path.basename(anchor_file))
        if not match:
            print(f"无法从文件名解析类别: {anchor_file}")
            continue
            
        category = match.group(1)
        print(f"提取的类别: {category}")
        
        # 查找对应的任务ID
        if category not in category_to_task:
            print(f"警告: 找不到类别 '{category}' 对应的任务ID")
            continue
            
        task_id = category_to_task[category]
        eval_file = os.path.join(eval_dir, f'eval_metrices_task{task_id}.json')
        print(f"对应的评估文件: {eval_file}")
        
        if not os.path.exists(eval_file):
            print(f"错误: 评估文件不存在: {eval_file}")
            continue

        # 加载评估数据
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            print(f"成功加载评估数据，包含 {len(eval_data)} 个条目")
        except Exception as e:
            print(f"加载评估文件失败: {str(e)}")
            continue

        # 加载原始anchor数据
        try:
            with open(anchor_file, 'r') as f:
                anchor_data = json.load(f)
            print(f"成功加载anchor数据，包含 {len(anchor_data)} 个条目")
        except Exception as e:
            print(f"加载anchor文件失败: {str(e)}")
            continue

        # 更新anchor数据
        found_count = 0
        not_found_count = 0
        for item in anchor_data:
            result_id = item["result_id"]
            
            # 查找匹配的评估结果
            if result_id in eval_data:
                prediction = eval_data[result_id][0]["pred"]
                item["prediction"] = prediction
                found_count += 1
            else:
                print(f"警告: 结果ID {result_id} 在 {eval_file} 中未找到")
                not_found_count += 1
        
        # 创建新文件名（添加任务ID后缀）
        file_dir = os.path.dirname(anchor_file)
        file_base = os.path.basename(anchor_file)
        file_name, file_ext = os.path.splitext(file_base)
        
        # 检查文件名是否已经包含任务ID
        if not re.search(r'_\d+$', file_name):
            new_file_name = f"{file_name}_{task_id}{file_ext}"
        else:
            new_file_name = file_base
        
        new_anchor_file = os.path.join(file_dir, new_file_name)
        
        # 保存更新后的数据到新文件
        try:
            with open(new_anchor_file, 'w') as f:
                json.dump(anchor_data, f, indent=2)
            print(f"成功保存文件: {new_anchor_file} (类别: {category}, 任务ID: {task_id})")
            print(f"匹配结果: 找到 {found_count} 个, 未找到 {not_found_count} 个")
        except Exception as e:
            print(f"保存文件失败: {str(e)}")

    print("\n所有文件处理完成")