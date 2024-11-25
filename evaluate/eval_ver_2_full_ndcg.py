import datasets
import numpy as np

# init
test_ds_path = "test_retrieval_ja"
rank_txt_path = "/home/trung/EXPERIMENT_50_HARD_1BATCH_ROUND2_JP (BEST PROPOSE - ROUND2 USING LLM WITH JAPANESE)/rank_ensemble/merge_rank.txt"
ks = [3, 5, 10, 20, 50, 100, 200]  # Danh sách các k cần tính precision và recall

# Load lại test dataset
test_ds = datasets.load_from_disk(test_ds_path)

query_relate = {}
for i in range(len(test_ds['query_id'])):
    query_relate[test_ds['query_id'][i]] = []
    for pos in test_ds['positive_passages'][i]:
        query_relate[test_ds['query_id'][i]].append(pos['docid'])

top10_predict = {}
# load file rank.scifact.txt
with open(rank_txt_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split("\t")
        query_id = parts[0]
        doc_id = parts[1]
        if query_id not in top10_predict:
            top10_predict[query_id] = []
        top10_predict[query_id].append(doc_id)

# Nhập giá trị u từ người dùng
u = int(input("Nhập giá trị u để tính MRR@u và MAP@u: "))

# Tính precision@k và recall@k với k thuộc {1, 3, 5, 10}
precisions = {k: 0 for k in ks}
recalls = {k: 0 for k in ks}

for query_id in top10_predict:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    for k in ks:
        count_retrieval = 0
        for i in range(min(k, len(predict_passages))):
            if predict_passages[i] in positive_passages:
                count_retrieval += 1
        precisions[k] += count_retrieval / k
        recalls[k] += count_retrieval / len(positive_passages)

for k in ks:
    precisions[k] /= len(top10_predict)
    recalls[k] /= len(top10_predict)

print("Precisions@k: ", precisions)
print("Recalls@k: ", recalls)

# Tính MAP@u
APs = []
for query_id in top10_predict:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    count_retrieval = 0
    sum_precision = 0
    for i in range(min(u, len(predict_passages))):  # Chỉ xét top u dự đoán
        if predict_passages[i] in positive_passages:
            count_retrieval += 1
            sum_precision += count_retrieval / (i + 1)
    APs.append(sum_precision / len(positive_passages))

MAP = sum(APs) / len(APs)
print(f"MAP@{u}: ", MAP)

# Tính MRR@u
MRRs = []
for query_id in top10_predict:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    for i in range(min(u, len(predict_passages))):  # Chỉ xét top u dự đoán
        if predict_passages[i] in positive_passages:
            MRRs.append(1 / (i + 1))
            break

MRR = sum(MRRs) / len(MRRs)
print(f"MRR@{u}: ", MRR)

# Tính my_recall chia cho giá trị nhỏ nhất của u và tổng số tài liệu liên quan
my_recalls = {k: 0 for k in ks}

for query_id in top10_predict:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    for k in ks:
        count_retrieval = 0
        for i in range(min(k, len(predict_passages))):
            if predict_passages[i] in positive_passages:
                count_retrieval += 1
        my_recalls[k] += count_retrieval / min(k, len(positive_passages))

for k in ks:
    my_recalls[k] /= len(top10_predict)

print("My_recalls@k: ", my_recalls)

# Tính NDCG@10
ndcgs_10 = []
for query_id in top10_predict:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    dcg = 0.0
    for i in range(min(10, len(predict_passages))):  # Chỉ xét top 10 dự đoán
        if predict_passages[i] in positive_passages:
            relevance = 1  # Giả sử độ liên quan là 1 cho mỗi tài liệu liên quan
            dcg += relevance / np.log2(i + 2)  # i+2 vì i bắt đầu từ 0

    # Tính IDCG@10 (DCG lý tưởng với các tài liệu liên quan được xếp hạng đầu tiên)
    ideal_relevances = [1] * min(10, len(positive_passages))  # Các tài liệu liên quan lý tưởng đều có độ liên quan là 1
    idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances)])

    # Tính NDCG cho truy vấn hiện tại
    ndcg = dcg / idcg if idcg > 0 else 0
    ndcgs_10.append(ndcg)

# Tính NDCG trung bình trên toàn bộ tập truy vấn
avg_ndcg_10 = sum(ndcgs_10) / len(ndcgs_10)
print("NDCG@10: ", avg_ndcg_10)
