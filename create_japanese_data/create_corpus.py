import json
import random

corpus_data = "/home/trungquang/Japanese_retrieval_with_LLM_new_propose/vjdatabase/legal_corpus_12x7.json"

corpus = json.load(open(corpus_data, encoding='utf-8'))

# trước khi xử lý corpus
print(len(corpus))
print(corpus[0]['law_id'])
print(len(corpus[0]['articles']))

# chuyển thành format mới như sau:
# Dataset({
#     features: ['query_id', 'query', 'positive_passages', 'negative_passages'],
#     num_rows: 400782
# })

data_new = {
    "docid": [],
    "title": [], 
    "text": []
}

for law in corpus:
    for article in law['articles']: 
        data_new["docid"].append(law['law_id'] + "_" + str(article['article_id']))
        data_new["title"].append(article['title'])
        data_new["text"].append(article['text'])

print("Sau khi xử lý train")
print(len(data_new["docid"]))
print(data_new["docid"][0])
print(data_new["title"][0])
print(data_new["text"][0])

# Lưu lại dataset và load bằng load_dataset
import datasets
dataset = datasets.Dataset.from_dict(data_new)

# Lưu lại dataset
dataset.save_to_disk("./legal_corpus_retrieval_ja")

# Load lại dataset
dataset = datasets.load_from_disk("./legal_corpus_retrieval_ja")
print(dataset)

# print 1 dòng dữ liệu
print(dataset['docid'][0])
print(dataset['text'][0])
print(dataset['title'][0])