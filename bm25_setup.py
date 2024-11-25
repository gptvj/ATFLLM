import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
import argparse
from utils import bm25_tokenizer, calculate_f2

class Config:
    data_path = "vjdatabase"
    save_bm25 = "saved_model"
    top_k_bm25 = 50
    bm25_k1 = 0.4
    bm25_b = 0.6

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="vjdatabase", type=str)
    parser.add_argument("--load_docs", action="store_false")
    parser.add_argument("--num_eval", default=500, type=str)
    args = parser.parse_args()
    cfg = Config()
    
    save_path = cfg.save_bm25
    os.makedirs(save_path, exist_ok=True)

    raw_data = cfg.data_path
    corpus_path = os.path.join(raw_data, "legal_corpus_12x7.json")

    data = json.load(open(corpus_path, encoding = 'utf-8'))

    if args.load_docs:
        print("Process documents")
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
                    
                tokens = bm25_tokenizer(article_full)
                documents.append(tokens)
                doc_refers.append([law_id, article_id, article_full])
    
        
        with open(os.path.join(save_path, "documents_manual"), "wb") as documents_file:
            pickle.dump(documents, documents_file)
        with open(os.path.join(save_path,"doc_refers_saved"), "wb") as doc_refer_file:
            pickle.dump(doc_refers, doc_refer_file)
    else:
        with open(os.path.join(save_path, "documents_manual"), "rb") as documents_file:
            documents = pickle.load(documents_file)
        with open(os.path.join(save_path,"doc_refers_saved"), "rb") as doc_refer_file:
            doc_refers = pickle.load(doc_refer_file)
            


    train_path = os.path.join(raw_data, "train_12x7_retrieval.json")
    data = json.load(open(train_path, encoding = 'utf-8'))
    items = data["items"]
    print(len(items))

    print(len(documents))
    print(len(doc_refers))


    bm25 = BM25Plus(documents, k1=cfg.bm25_k1, b=cfg.bm25_b)
    with open(os.path.join(save_path, "bm25_Plus_04_06_model_full_manual_stopword"), "wb") as bm_file:
        pickle.dump(bm25, bm_file)
        