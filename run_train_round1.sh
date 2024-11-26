#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=1

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# main
python scripts/repllama/train.py --output_dir pretrained_model/model_repllama_50_hard_round1_2_batch --model_name_or_path meta-llama/Llama-2-7b-hf --save_steps 100 --dataset_name japanese_datasets/train_retrieval_ja --bf16 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --gradient_checkpointing --train_n_passages 15 --learning_rate 5e-4 --q_max_len 512 --p_max_len 512 --num_train_epochs 15 --logging_steps 100 --overwrite_output_dir --dataset_proc_num 32 --negatives_x_device --warmup_steps 100