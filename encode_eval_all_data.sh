#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=1

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Create directory for embeddings if it doesn't exist
mkdir -p ja_embedding_round2_1_bs_ne

# List of model checkpoints



# models=('model_repllama_50_hard_round1_2_batch/checkpoint-100' 'model_repllama_50_hard_round1_2_batch/checkpoint-200' 'model_repllama_50_hard_round1_2_batch/checkpoint-300' 'model_repllama_50_hard_round1_2_batch/checkpoint-400' 'model_repllama_50_hard_round1_2_batch/checkpoint-500' 'model_repllama_50_hard_round1_2_batch/checkpoint-600' 'model_repllama_50_hard_round1_2_batch/checkpoint-700' 'model_repllama_50_hard_round1_2_batch/checkpoint-800' 'model_repllama_50_hard_round1_2_batch/checkpoint-900' )

models=('model_repllama_50_hard_round2_1_BATCH_THOI/checkpoint-3600' 'model_repllama_50_hard_round2_1_BATCH_THOI/checkpoint-3700' 'model_repllama_50_hard_round2_1_BATCH_THOI/checkpoint-3800' 'model_repllama_50_hard_round2_1_BATCH_THOI/checkpoint-3900' 'model_repllama_50_hard_round2_1_BATCH_THOI/checkpoint-4000' 'model_repllama_50_hard_round2_1_BATCH_THOI/checkpoint-4060')

# Loop through each model checkpoint
for model in "${models[@]}"; do
  echo "Running with model: $model"

  # Encode query
  python examples/repllama/encode.py \
    --output_dir temp \
    --model_name_or_path "$model" \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --fp16 \
    --per_device_eval_batch_size 1 \
    --q_max_len 512 \
    --p_max_len 512 \
    --dataset_name test_retrieval_ja \
    --encoded_save_path ja_embedding_round2_1_bs_ne/queries_embedding_${model//\//_}.pkl \
    --encode_is_qry

  # Encode corpus
  python examples/repllama/encode.py \
    --output_dir temp \
    --model_name_or_path "$model" \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --fp16 \
    --per_device_eval_batch_size 1 \
    --q_max_len 512 \
    --p_max_len 512 \
    --dataset_name legal_corpus_retrieval_ja \
    --encoded_save_path ja_embedding_round2_1_bs_ne/corpus_embedding_${model//\//_}.pkl \
    --encode_num_shard 1 \
    --encode_shard_index 0

  # Perform search
  python -m tevatron.faiss_retriever \
    --query_reps ja_embedding_round2_1_bs_ne/queries_embedding_${model//\//_}.pkl \
    --passage_reps ja_embedding_round2_1_bs_ne/corpus_embedding_${model//\//_}.pkl \
    --depth 200 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ja_embedding_round2_1_bs_ne/rank_japanese_${model//\//_}.txt


done

