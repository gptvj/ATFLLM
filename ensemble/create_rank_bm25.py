import os
import json
import pickle
from tqdm import tqdm
from rank_bm25 import BM25Plus
import argparse
import pandas as pd
from utils import bm25_tokenizer

# Function to load JSON

def load_json(file_path):
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)["items"]

# Function to sort and save BM25 results

def sort_and_save_bm25_scores(full_rank, output_file):
    # Create DataFrame from BM25 scoring results
    bm25_data = pd.DataFrame(full_rank, columns=["query_id", "corpus_id", "bm25_score"])

    # Ensure query_id is integer to sort correctly
    bm25_data["query_id"] = bm25_data["query_id"].astype(int)

    # Sort by query_id ascending and bm25_score descending
    sorted_data = bm25_data.sort_values(by=["query_id", "bm25_score"], ascending=[True, False])

    # Save results to output file without header
    sorted_data.to_csv(output_file, sep="\t", index=False, header=False)

    print(f"Sorted file saved at {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="vjdatabase", type=str, help="Path to the data")
    parser.add_argument("--sorted_file", default="temp/sorted_bm25_rank.txt", type=str, help="Path to save bm25_rank_sorted.txt")
    args = parser.parse_args()

    # Step 1: Load data from legal_corpus.json
    corpus_path = os.path.join(args.data_path, "legal_corpus.json")
    corpus_data = json.load(open(corpus_path, encoding='utf-8'))

    # Step 2: Create dict to hold law article information
    legal_dict = {}
    documents = []
    doc_refers = []

    for law_article in tqdm(corpus_data):
        law_id = law_article["law_id"]
        articles = law_article["articles"]
        for sub_article in articles:
            article_id = str(sub_article["article_id"])
            article_title = sub_article["title"]
            article_text = sub_article["text"]
            concat_id = law_id + "_" + article_id

            if concat_id not in legal_dict:
                legal_dict[concat_id] = {"title": article_title, "text": article_text}
                article_full = article_title + " " + article_text
                tokens = bm25_tokenizer(article_full)

                documents.append(tokens)
                doc_refers.append([law_id, article_id, article_full])

    # Step 3: Initialize BM25 model with tokenized documents
    bm25 = BM25Plus(documents, k1=0.4, b=0.6)

    # Step 4: Load training data from test_retrieval_data.json
    train_path = os.path.join(args.data_path, "test_retrieval_data.json")
    training_items = load_json(train_path)

    # Step 5: Compute BM25 scores and append to full_rank
    full_rank = []
    count_query_id = 1

    for item in tqdm(training_items):
        question = item["question_full"]
        tokenized_query = bm25_tokenizer(question)
        predictions = bm25.get_scores(tokenized_query)

        query_id = count_query_id
        count_query_id += 1

        for index, corpus_id in enumerate(legal_dict.keys()):
            score = predictions[index]
            full_rank.append([query_id, str(corpus_id), score])  # query_id là số nguyên

    # Step 6: Sort and save to sorted_bm25_rank.txt
    sort_and_save_bm25_scores(full_rank, args.sorted_file)
