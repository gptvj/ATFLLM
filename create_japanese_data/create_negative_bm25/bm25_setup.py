import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Plus
import argparse
from utils import bm25_tokenizer

# Define a Config class to hold default parameters
class Config:
    def __init__(self, data_path, save_bm25, top_k_bm25, bm25_k1, bm25_b):
        self.data_path = data_path
        self.save_bm25 = save_bm25
        self.top_k_bm25 = top_k_bm25
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

if __name__ == '__main__':
    # Add command-line arguments
    parser = argparse.ArgumentParser(description="BM25 Retrieval Model")
    parser.add_argument("--data_path", type=str, default="vjdatabase", help="Path to the data directory")
    parser.add_argument("--corpus_file", type=str, default="legal_corpus.json", help="File corpus in the data directory")
    parser.add_argument("--train_file", type=str, default="train_retrieval_data.json", help="File train in the data directory")
    parser.add_argument("--save_bm25", type=str, default="temp/saved_model", help="Directory to save BM25 model")
    parser.add_argument("--top_k_bm25", type=int, default=50, help="Top k results for BM25")
    parser.add_argument("--bm25_k1", type=float, default=0.4, help="BM25 k1 parameter")
    parser.add_argument("--bm25_b", type=float, default=0.6, help="BM25 b parameter")
    parser.add_argument("--load_docs", action="store_false", help="Load documents flag")
    parser.add_argument("--num_eval", type=int, default=500, help="Number of evaluations")
    args = parser.parse_args()

    # Initialize Config with command-line parameters
    cfg = Config(args.data_path, args.save_bm25, args.top_k_bm25, args.bm25_k1, args.bm25_b)

    save_path = cfg.save_bm25
    os.makedirs(save_path, exist_ok=True)

    # Read data from the corpus file
    corpus_path = os.path.join(cfg.data_path, args.corpus_file)
    data = json.load(open(corpus_path, encoding='utf-8'))

    # Process documents if not already available
    if args.load_docs:
        print("Processing documents...")
        documents = []
        doc_refers = []
        
        for law_article in tqdm(data):
            law_id = law_article["law_id"]
            law_articles = law_article["articles"]
            
            for sub_article in law_articles:
                article_id = sub_article["article_id"]
                article_title = sub_article["title"]
                article_text = sub_article["text"]
                article_full = article_title + " " + article_text
                
                # Tokenize article text for BM25
                tokens = bm25_tokenizer(article_full)
                documents.append(tokens)
                doc_refers.append([law_id, article_id, article_full])
        
        # Save the processed documents to pickle files
        with open(os.path.join(save_path, "documents_manual"), "wb") as documents_file:
            pickle.dump(documents, documents_file)
        with open(os.path.join(save_path, "doc_refers_saved"), "wb") as doc_refer_file:
            pickle.dump(doc_refers, doc_refer_file)
    else:
        # Load documents and document references from pickle files
        with open(os.path.join(save_path, "documents_manual"), "rb") as documents_file:
            documents = pickle.load(documents_file)
        with open(os.path.join(save_path, "doc_refers_saved"), "rb") as doc_refer_file:
            doc_refers = pickle.load(doc_refer_file)

    # Load the training data from the JSON file
    train_path = os.path.join(cfg.data_path, args.train_file)
    data = json.load(open(train_path, encoding='utf-8'))
    items = data["items"]
    print(f"Number of items in train data: {len(items)}")

    print(f"Number of documents: {len(documents)}")
    print(f"Number of document references: {len(doc_refers)}")

    # Initialize the BM25 model
    bm25 = BM25Plus(documents, k1=cfg.bm25_k1, b=cfg.bm25_b)
    
    # Save the trained BM25 model
    bm25_model_path = os.path.join(save_path, "bm25_Plus_model")
    with open(bm25_model_path, "wb") as bm_file:
        pickle.dump(bm25, bm_file)

    print(f"BM25 model saved to {bm25_model_path}")
