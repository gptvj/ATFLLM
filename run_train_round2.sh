#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


python examples/repllama/train_round2.py --output_dir model_repllama_50_hard_round2_1_BATCH_THOI --model_name_or_path model_repllama_50_hard_round1_2_batch/checkpoint-2800 --tokenizer_name meta-llama/Llama-2-7b-hf --save_steps 100 --dataset_name hard_negative_ja_round1_for_training_round2 --bf16 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --gradient_checkpointing --train_n_passages 30 --learning_rate 5e-4 --q_max_len 512 --p_max_len 512 --num_train_epochs 20 --logging_steps 100 --overwrite_output_dir --dataset_proc_num 32 --negatives_x_device --warmup_steps 100

