#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Argument for model name or path and save folder
MODEL_NAME_OR_PATH=$1
SAVE_FOLDER=$2

# Create output directory
mkdir -p $SAVE_FOLDER

# Encode query
python scripts/repllama/encode.py --output_dir temp --model_name_or_path $MODEL_NAME_OR_PATH --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 2 --q_max_len 512 --p_max_len 512 --dataset_name japanese_datasets/test_retrieval_ja --encoded_save_path $SAVE_FOLDER/queries_embedding.pkl --encode_is_qry

# Encode corpus
python scripts/repllama/encode.py --output_dir temp --model_name_or_path $MODEL_NAME_OR_PATH --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 2 --q_max_len 512 --p_max_len 512 --dataset_name japanese_datasets/legal_corpus_retrieval_ja --encoded_save_path $SAVE_FOLDER/corpus_embedding.pkl --encode_num_shard 1 --encode_shard_index 0

# Search
python -m tevatron.faiss_retriever --query_reps $SAVE_FOLDER/queries_embedding.pkl --passage_reps $SAVE_FOLDER/corpus_embedding.pkl --depth 743 --batch_size 64 --save_text --save_ranking_to $SAVE_FOLDER/rank_japanese.txt
