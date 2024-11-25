#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=1

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


mkdir jp_round1_2800



# encode query

python examples/repllama/encode.py --output_dir temp --model_name_or_path model_repllama_50_hard_round1_2_batch/checkpoint-2800 --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 2 --q_max_len 512 --p_max_len 512 --dataset_name test_retrieval_ja --encoded_save_path jp_round1_2800/queries_embedding.pkl --encode_is_qry



# encode query - for training round 2
# python examples/repllama/encode.py --output_dir temp --model_name_or_path model_repllama_50_hard_round1_2_batch/checkpoint-1100 --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 8 --q_max_len 512 --p_max_len 512 --dataset_name train_retrieval_ja --encoded_save_path jp_round1_2800/queries_embedding.pkl --encode_is_qry > trung.log



# encode corpus

python examples/repllama/encode.py --output_dir temp --model_name_or_path model_repllama_50_hard_round1_2_batch/checkpoint-2800 --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 2 --q_max_len 512 --p_max_len 512 --dataset_name legal_corpus_retrieval_ja --encoded_save_path jp_round1_2800/corpus_embedding.pkl --encode_num_shard 1 --encode_shard_index 0





# Search
python -m tevatron.faiss_retriever --query_reps jp_round1_2800/queries_embedding.pkl --passage_reps 'jp_round1_2800/corpus_embedding.pkl' --depth 743 --batch_size 64 --save_text --save_ranking_to jp_round1_2800/rank_japanese.txt