import json
import random

train_data = "/home/trungquang/EXPERIMENT_50_HARD_2BATCH_JAPANESE/vjdatabase/validation_12x7_retrieval.json"
# dev_data = "./validation_retrieval_ms_marco.json"
corpus_data = "/home/trungquang/EXPERIMENT_50_HARD_2BATCH_JAPANESE/vjdatabase/legal_corpus_12x7.json"

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
    "positive_passages": []
}

# read file pickle chứa dữ liệu negative_passages
import pickle

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

print("Sau khi xử lý train")
print()

# Lưu lại dataset và load bằng load_dataset
import datasets
dataset = datasets.Dataset.from_dict(data_new)

# Lưu lại dataset
dataset.save_to_disk("./validation_retrieval_ja")

# Load lại dataset
dataset = datasets.load_from_disk("./validation_retrieval_ja")
print(dataset)

# print 1 dòng dữ liệu
print(dataset['query_id'][0])
print(dataset['query'][0])
print(dataset['positive_passages'][0])