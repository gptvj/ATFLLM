# Tạo corpus
python create_japanese_data/create_corpus.py --input_file ./vjdatabase/legal_corpus.json --output_folder ./japanese_datasets --output_name legal_corpus_retrieval_ja

# Tạo negative using BM25 for train dataset
python create_japanese_data/create_negative_bm25/bm25_setup.py --data_path ./vjdatabase --save_bm25 ./temp/saved_model --top_k_bm25 50 --bm25_k1 0.4 --bm25_b 0.6


# Tạo train dataset
python create_japanese_data/create_data.py --train_data ./vjdatabase/train_retrieval_data.json --corpus_data ./vjdatabase/legal_corpus.json --negative_data ./pair_data/bm_25_pairs_training_top50 --output_folder ./japanese_datasets --output_name train_retrieval_ja
