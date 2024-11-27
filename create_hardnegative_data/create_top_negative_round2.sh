#!/bin/bash

# Check if the parameter model_name_or_path is passed
if [ -z "$1" ]; then
    echo "You need to provide model_name_or_path as an argument."
    exit 1
fi

# Set CUDA device to 0
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Get model_name_or_path from the command-line argument
MODEL_NAME_OR_PATH=$1

mkdir -p temp/japanese_embedding_negative_round2

# Encode queries for round 2 from the train dataset
python scripts/repllama/encode.py --output_dir temp/encode --model_name_or_path $MODEL_NAME_OR_PATH --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 28 --q_max_len 512 --p_max_len 512 --dataset_name japanese_datasets/train_retrieval_ja --encoded_save_path temp/japanese_embedding_negative_round2/queries_embedding.pkl --encode_is_qry

# Encode corpus
python scripts/repllama/encode.py --output_dir temp/encode --model_name_or_path $MODEL_NAME_OR_PATH --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 28 --q_max_len 512 --p_max_len 512 --dataset_name japanese_datasets/legal_corpus_retrieval_ja --encoded_save_path temp/japanese_embedding_negative_round2/corpus_embedding.pkl --encode_num_shard 1 --encode_shard_index 0

# Perform search
python -m tevatron.faiss_retriever --query_reps temp/japanese_embedding_negative_round2/queries_embedding.pkl --passage_reps 'temp/japanese_embedding_negative_round2/corpus_embedding.pkl' --depth 200 --batch_size 64 --save_text --save_ranking_to temp/japanese_embedding_negative_round2/rank_japanese.txt
