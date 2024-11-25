import json
import random

train_data = "/home/trung/experiment_50_hardnegative_jp_llama2/vjdatabase/train_12x7_retrieval.json"

corpus_data = "/home/trung/experiment_50_hardnegative_jp_llama2/vjdatabase/legal_corpus_12x7.json"

corpus = json.load(open(corpus_data, encoding='utf-8'))
data = json.load(open(train_data, encoding='utf-8'))

print("Trước khi xử lý train")
# print(data)
print(len(data['items']))
print(data['items'][0])

def find_in_corpus(law_id, article_id):
    for item in corpus:
        if item['law_id'] == law_id:
            for article in item['articles']:
                if str(article['article_id']) == article_id:
                    return {
                        'docid': item['law_id'] + '_' + str(article['article_id']),
                        'title': article['title'],
                        'text': article['text']
                    }

# chuyển thành format mới như sau:
# Dataset({
#     features: ['query_id', 'query', 'positive_passages', 'negative_passages'],
#     num_rows: 400782
# })
data_new = {
    "query_id": [],
    "query": [],
    "positive_passages": [],
    "negative_passages": []
}

# read file pickle chứa dữ liệu negative_passages
import pickle

with open('/home/trung/experiment_50_hardnegative_jp_llama2/pair_data/bm_25_pairs_training_top50', 'rb') as f:
    data_negative = pickle.load(f)
    print(len(data_negative))
    print(data_negative[0])

query_id_count = 1
for item in data['items']:
    data_new['query_id'].append(str(query_id_count))
    query_id_count += 1
    data_new['query'].append(item['question_full'])
    this_pos = []
    for i in range(len(item['relevant_articles'])):
        # print(item['relevant_articles'][i]['law_id'], item['relevant_articles'][i]['article_id'])
        this_pos.append(find_in_corpus(item['relevant_articles'][i]['law_id'], item['relevant_articles'][i]['article_id']))
    data_new['positive_passages'].append(this_pos)
    # data_new['negative_passages'].append()
    # Tìm kiếm các bài viết không liên quan bằng random
    this_neg = []
    set_id = set([item['relevant_articles'][i]['law_id']+ item['relevant_articles'][i]['article_id'] for i in range(len(item['relevant_articles']))])
    # lấy danh sách law và article từ data_negative
    # tìm các law và article của item['question_full'] trong data_negative
    list_law = []
    list_article = []
    for item_negative in data_negative:
        if item_negative['question'] == item['question_full']:
            list_law.append(item_negative['docid'])
            list_article.append(item_negative['article_id'])
    
    # print(len(list_law), len(list_article))
    for i in range(len(list_law)):
        if list_law[i] + list_article[i] not in set_id:
            this_neg.append(find_in_corpus(list_law[i], list_article[i]))
            
    data_new['negative_passages'].append(this_neg)

print("Sau khi xử lý train")
print()

# Lưu lại dataset và load bằng load_dataset
import datasets
dataset = datasets.Dataset.from_dict(data_new)

# Lưu lại dataset
dataset.save_to_disk("./train_retrieval_ja")

# Load lại dataset
dataset = datasets.load_from_disk("./train_retrieval_ja")
print(dataset)

# print 1 dòng dữ liệu
i = 2
print(dataset['query_id'][i])
print(dataset['query'][i])
print(dataset['positive_passages'][i])
print(dataset['negative_passages'][i])
print(len(dataset['negative_passages'][i]))