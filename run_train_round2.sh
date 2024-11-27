#!/bin/bash

# Kiểm tra nếu model_name_or_path được truyền vào
if [ -z "$1" ]; then
    echo "Bạn cần cung cấp model_name_or_path làm đối số."
    exit 1
fi

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Lấy giá trị model_name_or_path từ tham số dòng lệnh
MODEL_NAME_OR_PATH=$1

# Chạy lệnh training với model_name_or_path là đối số
python scripts/repllama/train_round2.py --output_dir pretrained_model/model_repllama_50_hard_round2_1_BATCH_THOI --model_name_or_path $MODEL_NAME_OR_PATH --tokenizer_name meta-llama/Llama-2-7b-hf --save_steps 100 --dataset_name hard_negative_ja_round1_for_training_round2 --bf16 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --gradient_checkpointing --train_n_passages 30 --learning_rate 5e-4 --q_max_len 512 --p_max_len 512 --num_train_epochs 20 --logging_steps 100 --overwrite_output_dir --dataset_proc_num 32 --negatives_x_device --warmup_steps 100
