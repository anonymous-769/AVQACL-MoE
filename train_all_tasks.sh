#!/bin/bash
#SBATCH -J Test
#SBATCH -o logs/train_all_tasks.log
#SBATCH -e logs/train_all_tasks.err
#SBATCH -N 1 -n 1 -p GPU-8A100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=gpu_8a100
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH="/path/to/AVQACL-MoE:$PYTHONPATH"
cd /path/to/AVQACL-MoE
source activate unimoe

echo "Starting Anchor MoE Mode Training for All Tasks"
echo "================================================================================"

# 通用训练函数
train_task() {
    local task_id=$1
    local data_path=$2
    local output_dir=$3
    
    echo "Training Task $task_id..."
    
    export MASTER_PORT=9873
    
    deepspeed --num_gpus 1 --num_nodes 1 \
        --master_addr "localhost" --master_port $MASTER_PORT \
        AVQACL-MoE/train/train_anchor_moe.py \
        --deepspeed ./scripts/zero2.json \
        --model_name_or_path checkpoints/AVQACL-MoE-base-converted \
        --version v1 \
        --data_path "$data_path" \
        --output_dir "$output_dir" \
        --vision_tower checkpoints/clip-vit-large-patch14-336 \
        --audio_tower checkpoints/BEATs_iter3_plus_AS2M.pt \
        --freeze_backbone True \
        --tune_mm_mlp_adapter False \
        --tune_mm_audio_projector False \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --num_train_epochs 5 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 5 \
        --learning_rate 4e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 3 \
        --lazy_preprocess True \
        --report_to none \
        --llm_lora_enable False \
        --mix_va True \
        --moe_training_mode anchor \
        --anchor_save_path output/AVQA/anchors \
        --expert_weights_save_path output/AVQA/experts \
        --cur_task $task_id
    
    return $?
}

# 创建输出目录
mkdir -p output/AVQA/anchors
mkdir -p output/AVQA/experts
mkdir -p logs

# Train Task 0 (Come_From)
train_task 0 "data_sample/train_Come_From.json" "output/AVQA/AVQACL-MoE_task0_ckpt"
if [ $? -ne 0 ]; then
    echo "Error: Task 0 (Come_From) training failed!"
    exit 1
fi
echo "Task 0 (Come) training completed successfully."

# Train Task 1 (Happening)
train_task 1 "data_sample/train_Happening.json" "output/AVQA/AVQACL-MoE_task1_ckpt"
if [ $? -ne 0 ]; then
    echo "Error: Task 1 (Happening) training failed!"
    exit 1
fi
echo "Task 1 (Happening) training completed successfully."

# Train Task 2 (Where)
train_task 2 "data_sample/train_Where.json" "output/AVQA/AVQACL-MoE_task2_ckpt"
if [ $? -ne 0 ]; then
    echo "Error: Task 2 (Where) training failed!"
    exit 1
fi
echo "Task 2 (Where) training completed successfully."

# Train Task 3 (Which)
train_task 3 "data_sample/train_Which.json" "output/AVQA/AVQACL-MoE_task3_ckpt"
if [ $? -ne 0 ]; then
    echo "Error: Task 3 (Which) training failed!"
    exit 1
fi
echo "Task 3 (Which) training completed successfully."

echo "================================================================================"
echo "All task training completed successfully!"
echo "Expert weights saved in: output/AVQA/experts/"
echo "  - expert_task_0.pt"
echo "  - expert_task_1.pt"
echo "  - expert_task_2.pt"
echo "  - expert_task_3.pt"
echo "Anchors saved in: output/AVQA/anchors/"
echo "  - anchor_task_0.pt"
echo "  - anchor_task_1.pt"
echo "  - anchor_task_2.pt"
echo "  - anchor_task_3.pt"
echo "================================================================================"