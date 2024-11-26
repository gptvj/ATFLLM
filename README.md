## Step to Step

### Step 1: Create negative for round 1 using BM25
### Step 1: Create Japanese dataset from root data
- using create_japanese_data folder
*from ATFLLM*
1. using command to create legal_corpus_retrieval_ja: python create_japanese_data/create_corpus.py
2. using command to create train_retrieval_ja: python create_japanese_data/create_data.py
3. using command to create validation_retrieval_ja and test_retrieval_ja: python create_japanese_data/create_test.py
### Step 2: Training round 1
./run_train_round1.sh 
### Step 3: Create hard negative for round 2
### Step 4: Training round 2
### Step 5: Evalate
- using evaluate folder