#!/bin/bash
#SBATCH -J Test_Eval
#SBATCH -o logs/eval_CL_tasks.log
#SBATCH -e logs/eval_CL_tasks.err
#SBATCH -N 1 -n 1 -p GPU-8A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=gpu_8a100

export PYTHONPATH="/home/qukungroup/xiaoyao/myprojects/AVQACL-MoE:$PYTHONPATH"
cd /home/qukungroup/xiaoyao/myprojects/AVQACL-MoE
source activate unimoe

echo "Starting AVQACL-MoE Anchor Mode Evaluation for CL Tasks"
echo "================================================================================"

# 通用评估函数
eval_task() {
    local task_id=$1
    local data_path=$2
    local output_file=$3
    local num_experts=$4
    
    echo "Evaluating Task $task_id with $num_experts experts..."
    
    python AVQACL-MoE/eval_anchor.py \
        --data_path "$data_path" \
        --output "eval_results/AVQA_CL/${num_experts}_experts/$output_file" \
        --data_type "vqa" \
        --model_path "checkpoints/AVQACL-MoE-base-converted" \
        --num_experts $num_experts \
        --anchor_save_path "output/AVQA/anchors" \
        --expert_weights_save_path "output/AVQA/experts" \
        --use_anchor_mode
    
    return $?
}

# 定义任务列表
declare -A tasks=(
    [0]="data_sample/test_Come_From.json task0_come_results.json"
    [1]="data_sample/test_Happening.json task1_happening_results.json"
    [2]="data_sample/test_Where.json task2_where_results.json"
    [3]="data_sample/test_Which.json task3_which_results.json"
)

# 循环不同专家数量配置
for num_experts in {1..4}; do
    echo "Starting evaluation with $num_experts experts..."
    echo "-------------------------------------------------------------------------------"
    
    # 根据专家数量确定任务范围
    task_range=$((num_experts - 1))
    
    # 创建专家数量特定的结果目录
    mkdir -p "eval_results/AVQA_CL/${num_experts}_experts/"
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 执行对应范围内的任务
    for task_id in $(seq 0 $task_range); do
        # 检查任务ID是否有效
        if [ -z "${tasks[$task_id]}" ]; then
            echo "Warning: Task $task_id not defined! Skipping..."
            continue
        fi
        
        # 解析任务数据
        read -ra task_data <<< "${tasks[$task_id]}"
        data_path="${task_data[0]}"
        output_file="${task_data[1]}"
        
        # 执行评估
        eval_task $task_id "$data_path" "$output_file" $num_experts
        if [ $? -ne 0 ]; then
            echo "Error: Task $task_id evaluation failed with $num_experts experts!"
            exit 1
        fi
        echo "Task $task_id evaluation completed successfully with $num_experts experts."
    done
    
    # 计算并显示耗时
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "-------------------------------------------------------------------------------"
    echo "Evaluation with $num_experts experts completed in $duration seconds."
    echo "Results saved in: eval_results/AVQA_CL/${num_experts}_experts/"
    echo "================================================================================"
done

echo "All evaluations completed successfully!"