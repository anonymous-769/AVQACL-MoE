#!/bin/bash
#SBATCH -J Test_Eval
#SBATCH -o logs/eval_all_tasks.log
#SBATCH -e logs/eval_all_tasks.err
#SBATCH -N 1 -n 1 -p GPU-8A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=gpu_8a100

export PYTHONPATH="/home/qukungroup/xiaoyao/myprojects/AVQACL-MoE:$PYTHONPATH"
cd /home/qukungroup/xiaoyao/myprojects/AVQACL-MoE
source activate unimoe

echo "Starting AVQACL-MoE Anchor Mode Evaluation for All Tasks"
echo "================================================================================"

# 通用评估函数
eval_task() {
    local task_id=$1
    local data_path=$2
    local output_file=$3
    
    echo "Evaluating Task $task_id..."
    
    python AVQACL-MoE/eval_anchor.py \
        --data_path "$data_path" \
        --output "eval_results/AVQA/$output_file" \
        --data_type "vqa" \
        --model_path "checkpoints/AVQACL-MoE-base-converted" \
        --num_experts 4 \
        --anchor_save_path "output/AVQA/anchors" \
        --expert_weights_save_path "output/AVQA/experts" \
        --use_anchor_mode
    
    return $?
}

# 创建输出目录
mkdir -p eval_results/AVQA/

# Evaluate Task 0 (Come_From)
eval_task 0 "data_sample/test_Come_From.json" "task0_come_results.json"
if [ $? -ne 0 ]; then
    echo "Error: Task 0 (Come_From) evaluation failed!"
    exit 1
fi
echo "Task 0 (Come_From) evaluation completed successfully."

# Evaluate Task 1 (Happening)
eval_task 1 "data_sample/test_Happening.json" "task1_happening_results.json"
if [ $? -ne 0 ]; then
    echo "Error: Task 1 (Happening) evaluation failed!"
    exit 1
fi
echo "Task 1 (Happening) evaluation completed successfully."

# Evaluate Task 2 (Where)
eval_task 2 "data_sample/test_Where.json" "task2_where_results.json"
if [ $? -ne 0 ]; then
    echo "Error: Task 2 (Where) evaluation failed!"
    exit 1
fi
echo "Task 2 (Where) evaluation completed successfully."

# Evaluate Task 3 (Which)
eval_task 3 "data_sample/test_Which.json" "task3_which_results.json"
if [ $? -ne 0 ]; then
    echo "Error: Task 3 (Which) evaluation failed!"
    exit 1
fi
echo "Task 3 (Which) evaluation completed successfully."

echo "================================================================================"
echo "All task evaluations completed successfully!"
echo "Results saved in eval_results/AVQA/ directory:"
echo "  - eval_results/AVQA/task0_come_results.json"
echo "  - eval_results/AVQA/task1_happening_results.json"
echo "  - eval_results/AVQA/task2_where_results.json"
echo "  - eval_results/AVQA/task3_which_results.json"
echo "================================================================================"