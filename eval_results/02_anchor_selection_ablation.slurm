#!/bin/bash
#SBATCH -J Test
#SBATCH -o logs/extract_log_anchor_select.log
#SBATCH -e logs/extract_log_anchor_select.err
#SBATCH -N 1 -n 1 -p CPU-64C256GB --qos=qos_cpu_64c256gb
#SBATCH --exclude=cnode73
#SBATCH --ntasks=6
#SBATCH --time=240:00:00
# 路由选择消融实验
cd /home/qukungroup/xiaoyao/myprojects/AVQACL-MoE-Music/eval_results

source activate unimoe

# 首先要提取eval的log文件中的所有样本专家选择情况,_4.log是包含全部专家全部任务的完整验证
python extract_log_anchor_select.py \
  --log_files ../logs/eval_CL_tasks_551410_4.log \
  --output_dir ./AVQA_MUSIC_CL/Case_study

# 然后要提取每个样本的预测结果，添加（append）到eval得到的json对应文件中
python extract_pred_to_log.py\
  --anchor_dir "./AVQA_MUSIC_CL/Case_study" \
  --eval_dir "./AVQA_MUSIC_CL/4_experts" \
  --output_dir "./AVQA_MUSIC_CL/Case_study"

# 计算指标
python anchor_selection_metrices.py\
  --anchor_dir "./AVQA_MUSIC_CL/Case_study" \
  --output_dir "./AVQA_MUSIC_CL/Analysis_Results"
