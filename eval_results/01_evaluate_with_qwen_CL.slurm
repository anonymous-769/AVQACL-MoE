#!/bin/bash
#SBATCH -J Test
#SBATCH -o logs/EvalwithQwen.log
#SBATCH -e logs/EvalwithQwen.err
#SBATCH -N 1 -n 1 -p CPU-64C256GB --qos=qos_cpu_64c256gb
#SBATCH --exclude=cnode73
#SBATCH --ntasks=6
#SBATCH --time=240:00:00
# CL必做的验证流程
cd /path/to/AVQACL-MoE-Music/eval_results

source activate unimoe

# current task 1
python evaluate_with_qwen_CL.py \
  --json_dir ./AVQA_MUSIC_CL/1_experts \
  --output_dir ./AVQA_MUSIC_CL/1_experts \
  --api_key sk-9a8a9da2358b4fdb9a5b1b4206b7498e \

# current task 2
python evaluate_with_qwen_CL.py \
  --json_dir ./AVQA_MUSIC_CL/2_experts \
  --output_dir ./AVQA_MUSIC_CL/2_experts \
  --api_key sk-9a8a9da2358b4fdb9a5b1b4206b7498e \

# current task 3
python evaluate_with_qwen_CL.py \
  --json_dir ./AVQA_MUSIC_CL/3_experts \
  --output_dir ./AVQA_MUSIC_CL/3_experts \
  --api_key sk-9a8a9da2358b4fdb9a5b1b4206b7498e \

# current task 4
python evaluate_with_qwen_CL.py \
  --json_dir ./AVQA_MUSIC_CL/4_experts \
  --output_dir ./AVQA_MUSIC_CL/4_experts \
  --api_key sk-9a8a9da2358b4fdb9a5b1b4206b7498e \

# current task 5
python evaluate_with_qwen_CL.py \
  --json_dir ./AVQA_MUSIC_CL/5_experts \
  --output_dir ./AVQA_MUSIC_CL/5_experts \
  --api_key sk-9a8a9da2358b4fdb9a5b1b4206b7498e \

python metrices_CL.py --results_dir "./AVQA_MUSIC_CL" --total_tasks 5