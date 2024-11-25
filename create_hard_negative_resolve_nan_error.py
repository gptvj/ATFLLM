import datasets

# init
test_ds_path = "train_retrieval_ja"
legal_ds_path = "legal_corpus_retrieval_ja"
rank_txt_path = "japanese_embedding_negative_round2/rank_japanese.txt"
save_path = "hard_negative_ja_round1_for_training_round2"

n = 50   # số lượng câu hard negative mỗi query

# Load lại test dataset
test_ds = datasets.load_from_disk(test_ds_path)
legal_ds = datasets.load_from_disk(legal_ds_path)

# Chuyển legal_ds thành dict để tra cứu nhanh hơn
legal_dict = {doc_id: {"docid": doc_id, "title": title, "text": text}
              for doc_id, title, text in zip(legal_ds['docid'], legal_ds['title'], legal_ds['text'])}

# Load top100_predict từ file
top100_predict = {}
is_NaN_score = {}
with open(rank_txt_path, "r", encoding='utf-8') as f: 
    for line in f:
        query_id, doc_id, score = line.strip().split("\t")[:3]
        score = float(score)
        top100_predict.setdefault(query_id, []).append(doc_id)
        if query_id not in is_NaN_score:
            is_NaN_score[query_id] = True if score < 0 else False

# Tạo hard negative dataset
new_hard_negative_dataset = {
    "query_id": [],
    "query": [],
    "positive_passages": [],
    "negative_passages": []
}

count_NaN = 0

for i, query_id in enumerate(test_ds['query_id']):
    positive_passages = test_ds['positive_passages'][i]
    predict_passages = top100_predict.get(query_id, [])

    new_hard_negative_dataset['query_id'].append(query_id)
    new_hard_negative_dataset['query'].append(test_ds['query'][i])
    new_hard_negative_dataset['positive_passages'].append(positive_passages)
    
    # thêm hard negative passages
    list_pos_id = {pos['docid'] for pos in positive_passages}
    nega_this = []
    if is_NaN_score[query_id]: 
        # lấy negative trong tập test_ds
        print("NaN")
        count_NaN += 1
        # print(test_ds['negative_passages'][i])
        # print(len(test_ds['negative_passages'][i]))
        # print(test_ds['negative_passages'][i][0])
        nega_this = [nega for nega in test_ds['negative_passages'][i]][:n]
    else: 
        nega_this = [legal_dict[doc_id] for doc_id in predict_passages if doc_id not in list_pos_id][:n]
    

    new_hard_negative_dataset['negative_passages'].append(nega_this)

print("count_NaN", count_NaN)

# Lưu lại dataset và load bằng load_dataset
dataset = datasets.Dataset.from_dict(new_hard_negative_dataset)
dataset.save_to_disk(save_path)

# Load lại dataset
dataset = datasets.load_from_disk(save_path)
print(dataset)
print("Done")
