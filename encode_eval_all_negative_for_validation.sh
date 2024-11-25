#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=1

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Create directory for embeddings if it doesn't exist
mkdir -p ja_validation_embedding

# List of model checkpoints



# models=('model_repllama_50_hard_round1_2_batch/checkpoint-100' 'model_repllama_50_hard_round1_2_batch/checkpoint-200' 'model_repllama_50_hard_round1_2_batch/checkpoint-300' 'model_repllama_50_hard_round1_2_batch/checkpoint-400' 'model_repllama_50_hard_round1_2_batch/checkpoint-500' 'model_repllama_50_hard_round1_2_batch/checkpoint-600' 'model_repllama_50_hard_round1_2_batch/checkpoint-700' 'model_repllama_50_hard_round1_2_batch/checkpoint-800' 'model_repllama_50_hard_round1_2_batch/checkpoint-900' )

models=('model_repllama_50_hard_round1_2_batch/checkpoint-100' 'model_repllama_50_hard_round1_2_batch/checkpoint-200' 'model_repllama_50_hard_round1_2_batch/checkpoint-300' 'model_repllama_50_hard_round1_2_batch/checkpoint-400' 'model_repllama_50_hard_round1_2_batch/checkpoint-500' 
'model_repllama_50_hard_round1_2_batch/checkpoint-600' 'model_repllama_50_hard_round1_2_batch/checkpoint-700' 'model_repllama_50_hard_round1_2_batch/checkpoint-800' 'model_repllama_50_hard_round1_2_batch/checkpoint-900' 'model_repllama_50_hard_round1_2_batch/checkpoint-1000' 'model_repllama_50_hard_round1_2_batch/checkpoint-1100' 'model_repllama_50_hard_round1_2_batch/checkpoint-1200' 'model_repllama_50_hard_round1_2_batch/checkpoint-1300' 'model_repllama_50_hard_round1_2_batch/checkpoint-1400' 'model_repllama_50_hard_round1_2_batch/checkpoint-1500' 'model_repllama_50_hard_round1_2_batch/checkpoint-1600' 'model_repllama_50_hard_round1_2_batch/checkpoint-1700' 'model_repllama_50_hard_round1_2_batch/checkpoint-1800' 'model_repllama_50_hard_round1_2_batch/checkpoint-1900' 'model_repllama_50_hard_round1_2_batch/checkpoint-2000' 
'model_repllama_50_hard_round1_2_batch/checkpoint-2100' 'model_repllama_50_hard_round1_2_batch/checkpoint-2200' 'model_repllama_50_hard_round1_2_batch/checkpoint-2300' 'model_repllama_50_hard_round1_2_batch/checkpoint-2400' 'model_repllama_50_hard_round1_2_batch/checkpoint-2500' 'model_repllama_50_hard_round1_2_batch/checkpoint-2600' 'model_repllama_50_hard_round1_2_batch/checkpoint-2700' 'model_repllama_50_hard_round1_2_batch/checkpoint-2800' 'model_repllama_50_hard_round1_2_batch/checkpoint-2900' 'model_repllama_50_hard_round1_2_batch/checkpoint-3000' 'model_repllama_50_hard_round1_2_batch/checkpoint-3045' )

# Loop through each model checkpoint
for model in "${models[@]}"; do
  echo "Running with model: $model"

  # Encode query
  python examples/repllama/encode.py \
    --output_dir temp \
    --model_name_or_path "$model" \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --fp16 \
    --per_device_eval_batch_size 26 \
    --q_max_len 512 \
    --p_max_len 512 \
    --dataset_name validation_retrieval_ja \
    --encoded_save_path ja_validation_embedding/queries_embedding_${model//\//_}.pkl \
    --encode_is_qry

  # Encode corpus
  python examples/repllama/encode.py \
    --output_dir temp \
    --model_name_or_path "$model" \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --fp16 \
    --per_device_eval_batch_size 26 \
    --q_max_len 512 \
    --p_max_len 512 \
    --dataset_name legal_corpus_retrieval_ja \
    --encoded_save_path ja_validation_embedding/corpus_embedding_${model//\//_}.pkl \
    --encode_num_shard 1 \
    --encode_shard_index 0

  # Perform search
  python -m tevatron.faiss_retriever \
    --query_reps ja_validation_embedding/queries_embedding_${model//\//_}.pkl \
    --passage_reps ja_validation_embedding/corpus_embedding_${model//\//_}.pkl \
    --depth 200 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ja_validation_embedding/rank_japanese_${model//\//_}.txt


done

