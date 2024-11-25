import json
import os
import re
from tqdm import tqdm
import argparse


def load_json(corpus_path):
    data = json.load(open(corpus_path, encoding='utf-8'))
    return data["items"]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="vjdatabase",
                        type=str, help="path to training data")
    parser.add_argument("--save_dir", default="./generated_data",
                        type=str, help="path to training data")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)


    corpus_path = os.path.join(args.data_dir, "legal_corpus_12x7.json")

    data = json.load(open(corpus_path, encoding='utf-8'))


    save_dict = {}

    count = 0
    for law_article in tqdm(data):
        law_id = law_article["law_id"]
        law_articles = law_article["articles"]

        for sub_article in law_articles:
            article_id = str(sub_article["article_id"])
            article_title = sub_article["title"]
            article_text = sub_article["text"]


            concat_id = law_id + "_" + article_id
            if concat_id not in save_dict:
                count += 1
                save_dict[concat_id] = {
                    "title": article_title, "text": article_text}

    print(count)
    # exit()
    print("Create legal dict from raw data")
    with open(os.path.join(args.save_dir, "legal_dict.json"), "w") as outfile:
        json.dump(save_dict, outfile)
    print("Finish")

