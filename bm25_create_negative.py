import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
from utils import bm25_tokenizer, load_json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_pair", default=50, type=int)
    parser.add_argument(
        "--model_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--data_path", default="vjdatabase",
                        type=str, help="path to input data")
    parser.add_argument("--save_pair_path", default="pair_data/",
                        type=str, help="path to save pair sentence directory")
    args = parser.parse_args()

    train_path = os.path.join(args.data_path, "train_12x7_retrieval.json")
    training_items = load_json(train_path)["items"]

    with open(args.model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    with open("saved_model/doc_refers_saved", "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    doc_data = json.load(
        open(os.path.join("generated_data", "legal_dict.json")))

    save_pairs = []
    top_n = args.top_pair
    for idx, item in tqdm(enumerate(training_items)):

        question = item["question_full"]                        
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)

        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)

        predictions = np.argpartition(
            doc_scores, len(doc_scores) - top_n)[-top_n:]

        # Save positive pairs
        for article in relevant_articles:
            save_dict = {}
            save_dict["question"] = question
            concat_id = article["law_id"] + "_" + article["article_id"]
            if (concat_id in doc_data.keys()):
                save_dict["document"] = doc_data[concat_id]["title"] + \
                    " " + doc_data[concat_id]["text"]
                save_dict["relevant"] = 1
                # save_pairs.append(save_dict)

        # Save negative pairs
        for idx, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]

            check = 0
            for article in relevant_articles:
                if str(pred[0]) == str(article["law_id"]) and str(pred[1]) == str(article["article_id"]):
                    check += 1

            if check == 0:
                save_dict = {}
                save_dict["question"] = question

                save_dict["docid"] = str(pred[0])
                save_dict["article_id"] = str(pred[1])

                concat_id = pred[0] + "_" + str(pred[1])
                save_dict["document"] = doc_data[concat_id]["title"] + \
                    " " + doc_data[concat_id]["text"]
                save_dict["relevant"] = 0
                save_pairs.append(save_dict)

    save_path = args.save_pair_path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"bm_25_pairs_training_top{top_n}"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)


