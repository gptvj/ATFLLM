import argparse
import json
import pickle
import os
from datasets import Dataset

# Hàm tìm kiếm trong corpus
def find_in_corpus(law_id, article_id, corpus):
    for item in corpus:
        if item['law_id'] == law_id:
            for article in item['articles']:
                if str(article['article_id']) == article_id:
                    return {
                        'docid': item['law_id'] + '_' + str(article['article_id']),
                        'title': article['title'],
                        'text': article['text']
                    }

# Hàm xử lý corpus và dữ liệu train
def process_data(train_data, corpus_data, negative_data, output_folder, output_name):
    # Load dữ liệu
    corpus = json.load(open(corpus_data, encoding='utf-8'))
    data = json.load(open(train_data, encoding='utf-8'))

    # Chuyển đổi dữ liệu sang định dạng mới
    data_new = {
        "query_id": [],
        "query": [],
        "positive_passages": [],
        "negative_passages": []
    }

    with open(negative_data, 'rb') as f:
        data_negative = pickle.load(f)

    query_id_count = 1
    for item in data['items']:
        data_new['query_id'].append(str(query_id_count))
        query_id_count += 1
        data_new['query'].append(item['question_full'])
        
        # Lấy danh sách positive passages
        this_pos = []
        for i in range(len(item['relevant_articles'])):
            this_pos.append(find_in_corpus(item['relevant_articles'][i]['law_id'], item['relevant_articles'][i]['article_id'], corpus))
        data_new['positive_passages'].append(this_pos)

        # Lấy danh sách negative passages
        this_neg = []
        set_id = set([item['relevant_articles'][i]['law_id']+ item['relevant_articles'][i]['article_id'] for i in range(len(item['relevant_articles']))])
        
        # Tìm các bài viết không liên quan trong dữ liệu negative
        list_law = []
        list_article = []
        for item_negative in data_negative:
            if item_negative['question'] == item['question_full']:
                list_law.append(item_negative['docid'])
                list_article.append(item_negative['article_id'])

        for i in range(len(list_law)):
            if list_law[i] + list_article[i] not in set_id:
                this_neg.append(find_in_corpus(list_law[i], list_article[i], corpus))
            
        data_new['negative_passages'].append(this_neg)

    # Tạo dataset từ dict
    dataset = Dataset.from_dict(data_new)

    # Tạo thư mục nếu chưa tồn tại và lưu dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, output_name)
    dataset.save_to_disk(output_path)
    
    print(f"Dataset created successfully at {output_path}")

    # Load lại dataset để kiểm tra
    dataset = Dataset.load_from_disk(output_path)
    return dataset

if __name__ == "__main__":
    # Parse tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Process legal corpus and training data into custom dataset format.")
    
    # Các tham số dòng lệnh
    parser.add_argument('--train_data', type=str, default="./vjdatabase/train_retrieval_data.json", help="Path to the train JSON file.")
    parser.add_argument('--corpus_data', type=str, default="./vjdatabase/legal_corpus.json", help="Path to the corpus JSON file.")
    parser.add_argument('--negative_data', type=str, default="./pair_data/bm_25_pairs_training_top50", help="Path to the negative passages pickle file.")
    parser.add_argument('--output_folder', type=str, default="./japanese_datasets", help="Folder to save the processed dataset.")
    parser.add_argument('--output_name', type=str, default="train_retrieval_ja", help="Name of the output dataset.")
    
    args = parser.parse_args()

    # Gọi hàm xử lý với tham số dòng lệnh
    dataset = process_data(args.train_data, args.corpus_data, args.negative_data, args.output_folder, args.output_name)

    # In thử 1 dòng dữ liệu
    i = 2
    print(f"query_id: {dataset['query_id'][i]}")
    print(f"query: {dataset['query'][i]}")
    print(f"positive_passages: {dataset['positive_passages'][i]}")
    print(f"negative_passages: {dataset['negative_passages'][i]}")
    print(f"Length of negative_passages: {len(dataset['negative_passages'][i])}")
