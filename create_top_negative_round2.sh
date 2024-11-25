#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


mkdir japanese_embedding_negative_round2



# encode query for round 2 from train_datasets

python examples/repllama/encode.py --output_dir temp --model_name_or_path model_repllama_50_hard_round1_2_batch/checkpoint-2800 --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 28 --q_max_len 512 --p_max_len 512 --dataset_name train_retrieval_ja --encoded_save_path japanese_embedding_negative_round2/queries_embedding.pkl --encode_is_qry




# encode corpus

python examples/repllama/encode.py --output_dir temp --model_name_or_path model_repllama_50_hard_round1_2_batch/checkpoint-2800 --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 28 --q_max_len 512 --p_max_len 512 --dataset_name legal_corpus_retrieval_ja --encoded_save_path japanese_embedding_negative_round2/corpus_embedding.pkl --encode_num_shard 1 --encode_shard_index 0





# Search
python -m tevatron.faiss_retriever --query_reps japanese_embedding_negative_round2/queries_embedding.pkl --passage_reps 'japanese_embedding_negative_round2/corpus_embedding.pkl' --depth 200 --batch_size 64 --save_text --save_ranking_to japanese_embedding_negative_round2/rank_japanese.txt