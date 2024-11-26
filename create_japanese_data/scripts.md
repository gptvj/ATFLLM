# Tạo corpus
python create_japanese_data/create_corpus.py --input_file ./vjdatabase/legal_corpus.json --output_folder ./japanese_datasets --output_name legal_corpus_retrieval_ja

# Tạo negative using BM25 for train dataset
python create_japanese_data/create_negative_bm25/bm25_setup.py --data_path ./vjdatabase --save_bm25 ./temp/saved_model --top_k_bm25 50 --bm25_k1 0.4 --bm25_b 0.6

# Tạo legal dict cho BM25
python create_japanese_data/create_negative_bm25/bm25_create_corpus.py --data_dir ./vjdatabase --corpus_file legal_corpus.json --save_dir temp/generated_data --save_file legal_dict.json

# Tạo negative pair top 50
python create_japanese_data/create_negative_bm25/bm25_create_negative.py --top_pair 50 --model_path ./temp/saved_model/bm25_Plus_model --data_path vjdatabase --save_pair_path ./temp/pair_data/


# Tạo train dataset
python create_japanese_data/create_data.py --train_data "vjdatabase/train_retrieval_data.json" --corpus_data "vjdatabase/legal_corpus.json" --negative_passages "temp/pair_data/bm_25_pairs_training_top50" --output_dir "japanese_datasets/train_retrieval_ja"

# Tạo validation, test dataset
python create_japanese_data/create_test.py --train_data "vjdatabase/validation_retrieval_data.json" --corpus_data "vjdatabase/legal_corpus.json" --output_dir "japanese_datasets/validation_retrieval_ja"

python create_japanese_data/create_test.py --train_data "vjdatabase/test_retrieval_data.json" --corpus_data "vjdatabase/legal_corpus.json" --output_dir "japanese_datasets/test_retrieval_ja"

