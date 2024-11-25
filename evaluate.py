# Tính precision, recall, map
# Cụ thể tính: precision@k và recall@k với k thuộc {1, 2, 3, 5, 7, 10}
# Tính my_recall thay vì chia cho tổng số tài liệu liên quan thì hãy chia cho giá trị nhỏ nhất của k và tổng số tài liệu liên quan
import datasets
from datasets import load_dataset

# init
# dataset_name_or_path = r".\test_retrieval_ms_marco"
# dataset_name_or_path = r"Tevatron/beir-corpus:scifact"
# dataset_name_or_path = r"Tevatron/scifact/train"

dataset_name_or_path = "test_retrieval_ja"




dataset_name = "/".join(dataset_name_or_path.split('/')[:-1])   
dataset_language = 'default'
info = dataset_name_or_path.split('/')
dataset_split = info[-1] if len(info) == 3 else 'train'
print("dataset_split", dataset_split)
is_hug = False
rank_txt_path = "/home/trung/EXPERIMENT_50_HARD_1BATCH_ROUND2_JP (BEST PROPOSE - ROUND2 USING LLM WITH JAPANESE)/rank_ensemble/merge_rank.txt"




ks = [3, 5, 10, 20, 50, 100, 200]    # Danh sách các k cần tính precision và recall

# Load lại test dataset
if is_hug:
    # test_ds = load_dataset(path = dataset_name_or_path,
    #                        name = dataset_language,
    #                     trust_remote_code=True)[dataset_split]
    test_ds = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)

else: 
    test_ds = datasets.load_from_disk(dataset_name_or_path)

# print(test_ds)
# for i in range(len(test_ds['query_id'])):
#     print(test_ds['query_id'][i])

# # chuyển t
# for i in range(len(test_ds['query_id'])):
#     if test_ds['query_id'][i] == '1088':
#         print(test_ds['query_id'][i])
#         print(test_ds['query'][i])
#         break

# for i in range(len(test_ds['docid'])):
#     if test_ds['docid'][i] == '37549932':
#         print(test_ds['docid'][i])
#         print(test_ds['title'][i])
#         print(test_ds['text'][i])
#         break

    # PGE 2 promotes intestinal tumor growth by altering the expression of tumor suppressing and DNA repair genes.
query_relate = {}

for i in range(len(test_ds['query_id'])): 
    # print(test_ds['query_id'][i])
    # print(test_ds['query'][i])
    # print(test_ds['positive_passages'][i])
    # print(test_ds['negative_passages'][i])
    query_relate[test_ds['query_id'][i]] = []
    for pos in test_ds['positive_passages'][i]: 
        query_relate[test_ds['query_id'][i]].append(pos['docid'])
    # break

# print(query_relate)

top10_predict = {}

# load file rank.scifact.txt
with open(rank_txt_path, "r", encoding='utf-8') as f: 
    lines = f.readlines()
    for line in lines: 
        parts = line.strip().split("\t")
        query_id = parts[0]
        doc_id = parts[1]
        if query_id not in top10_predict: 
            top10_predict[query_id] = []
        top10_predict[query_id].append(doc_id)

# print(top10_predict)

# in ra 5 query đầu tiên của query_relate và top10_predict
count = 0
for query_id in query_relate: 
    print("Query id: ", query_id)
    print("Query: ", test_ds['query'][count])
    print("Positive passages: ", query_relate[query_id])
    print("Top 10 predict: ", top10_predict[query_id])
    count += 1
    if count == 5: 
        break

# Tính precision@k và recall@k với k thuộc {1, 3, 5, 10}

precisions = {}
recalls = {}
for k in ks: 
    precisions[k] = 0
    recalls[k] = 0

for query_id in query_relate:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    for k in ks: 
        count_retrieval = 0
        for i in range(k): 
            if predict_passages[i] in positive_passages: 
                count_retrieval += 1
        precisions[k] += count_retrieval/k
        recalls[k] += count_retrieval/len(positive_passages)

for k in ks:
    precisions[k] /= len(query_relate)
    recalls[k] /= len(query_relate)

print("Precisions@k: ", precisions)
print("Recalls@k: ", recalls)

# Tính MAP
APs = []
for query_id in query_relate:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    count_retrieval = 0
    sum_precision = 0
    for i in range(len(predict_passages)): 
        if predict_passages[i] in positive_passages: 
            count_retrieval += 1
            sum_precision += count_retrieval/(i+1)
    APs.append(sum_precision/len(positive_passages))

MAP = sum(APs)/len(APs)
print("MAP@10: ", MAP)

# Tính my_recall thay vì chia cho tổng số tài liệu liên quan thì hãy chia cho giá trị nhỏ nhất của k và tổng số tài liệu liên quan
my_recalls = {}
for k in ks: 
    my_recalls[k] = 0

for query_id in query_relate:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    for k in ks: 
        count_retrieval = 0
        for i in range(k): 
            if predict_passages[i] in positive_passages: 
                count_retrieval += 1
        my_recalls[k] += count_retrieval/min(k, len(positive_passages))

for k in ks:
    my_recalls[k] /= len(query_relate)

print("My_recalls@k: ", my_recalls)
