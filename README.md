## Step to Step

### Step 1: Create negative for round 1 using BM25
### Step 1: Create Japanese dataset from root data
- using create_japanese_data folder
*from ATFLLM*
1. using command to create legal_corpus_retrieval_ja: 
python create_japanese_data/create_corpus.py
2. using command to create train_retrieval_ja: python create_japanese_data/create_data.py
3. using command to create validation_retrieval_ja and test_retrieval_ja: python create_japanese_data/create_test.py
### Step 2: Training round 1
./run_train_round1.sh 
### Step 3: Create hard negative for round 2
1. Tạo file rank từ model đã train ở round 1: create_hardnegative_data/create_top_negative_round2.sh
2. Loại bỏ NaN và thay thế bằng negative BM25 để tạo bộ data cho training round 2: python create_hardnegative_data/create_hard_negative_resolve_nan_error.py
### Step 4: Training round 2
./run_train_round2.sh
### Step 5: Evaluate
python evaluate_metrics.py --test_ds_path japanese_datasets/test_retrieval_ja --rank_txt_path temp/jp_round1_2800/rank_japanese.txt
python evaluate_metrics.py --test_ds_path japanese_datasets/test_retrieval_ja --rank_txt_path temp/jp_full_cp_ckpt_round2_400/rank_japanese.txt
### Step 6: Ensemble
python evaluate_metrics.py --test_ds_path japanese_datasets/test_retrieval_ja --rank_txt_path temp/jp_round1_2800/rank_japanese.txt