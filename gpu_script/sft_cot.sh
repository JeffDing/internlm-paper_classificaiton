#!/bin/bash

LOG_DIR="logs"
mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/internlm2_5-1_8b_lora_sft_cot_${TIMESTAMP}.log"

#export NPROC_PER_NODE=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


swift sft \
    --model "/tmp/code/swift_output/InternLM2.5-1.8B-Lora/v0-20250724-061112/checkpoint-10000-merged" \
    --train_type lora \
    --dataset '/tmp/code/dataset/new_output-2048-8K.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.1 \
    --split_dataset_ratio 0 \
    --lora_rank 128 \
    --lora_alpha 512 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --save_steps 500 \
    --save_total_limit 5 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir "./swift_output/InternLM2.5-1.8B-Lora-SFT-COT" \
    --dataloader_num_workers 256 \
    --attn_impl flash_attn \
    --model_author JeffDing \
    --model_name InternLM2.5-1.8B-Lora-SFT \
    > "$LOG_FILE" 2>&1 &

echo "Training started with PID $!"
echo "Log file: $LOG_FILE"
echo "To view logs in real-time, use:"
echo "tail -f $LOG_FILE"