import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Plus
from utils import bm25_tokenizer, load_json
import argparse

if __name__ == '__main__':
    # Add command-line arguments
    parser = argparse.ArgumentParser(description="Generate Positive and Negative Pairs for BM25 Retrieval")
    parser.add_argument("--top_pair", type=int, default=50, help="Top N relevant document pairs")
    parser.add_argument("--model_path", type=str, default="./temp/saved_model/bm25_Plus_model", help="Path to the BM25 model")
    parser.add_argument("--data_path", type=str, default="vjdatabase", help="Path to the input data directory")
    parser.add_argument("--train_file", type=str, default="train_retrieval_data.json", help="File train in the data directory")
    parser.add_argument("--doc_data_path", type=str, default="temp/generated_data/legal_dict.json", help="File to save the legal dict")
    parser.add_argument("--doc_refers_path", type=str, default="temp/saved_model/doc_refers_saved", help="File to save the doc refer")
    parser.add_argument("--save_pair_path", type=str, default="./temp/pair_data/", help="Directory to save the generated pairs")
    
    args = parser.parse_args()

    # Load the training data from the JSON file
    train_path = os.path.join(args.data_path, args.train_file)
    training_items = load_json(train_path)["items"]

    # Load the pre-trained BM25 model
    with open(args.model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)

    # Load the saved document references
    with open(args.doc_refers_path, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    # Load the document data from the JSON file
    doc_data = json.load(open(args.doc_data_path))

    save_pairs = []
    top_n = args.top_pair
    for idx, item in tqdm(enumerate(training_items)):
        question = item["question_full"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)

        # Tokenize the question
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)

        # Predict the most relevant documents
        predictions = np.argpartition(doc_scores, len(doc_scores) - top_n)[-top_n:]

        # Save the positive pairs
        # for article in relevant_articles:
        #     save_dict = {}
        #     save_dict["question"] = question
        #     save_dict["docid"] = str(article["law_id"])
        #     save_dict["article_id"] = str(article["article_id"])
        #     concat_id = article["law_id"] + "_" + article["article_id"]
        #     if concat_id in doc_data.keys():
        #         save_dict["document"] = doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]
        #         save_dict["relevant"] = 1
                # save_pairs.append(save_dict)

        # Save the negative pairs
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
                save_dict["document"] = doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]
                save_dict["relevant"] = 0
                save_pairs.append(save_dict)

    # Save the generated pairs to the specified directory
    save_path = args.save_pair_path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"bm_25_pairs_training_top{top_n}"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)

    print(f"Generated pairs saved to {save_path}")
