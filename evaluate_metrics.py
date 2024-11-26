import datasets
import numpy as np
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Evaluation script for retrieval metrics.')
parser.add_argument('--test_ds_path', type=str, required=True, help='Path to the test dataset')
parser.add_argument('--rank_txt_path', type=str, required=True, help='Path to the ranking text file')
args = parser.parse_args()

# Init
ks = [3, 5, 10, 20, 50, 100, 200]  # List of k values for precision and recall calculation

# Load the test dataset
test_ds = datasets.load_from_disk(args.test_ds_path)

query_relate = {}
for i in range(len(test_ds['query_id'])):
    query_relate[test_ds['query_id'][i]] = []
    for pos in test_ds['positive_passages'][i]:
        query_relate[test_ds['query_id'][i]].append(pos['docid'])

top10_predict = {}
# Load the rank file
with open(args.rank_txt_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split("\t")
        query_id = parts[0]
        doc_id = parts[1]
        if query_id not in top10_predict:
            top10_predict[query_id] = []
        top10_predict[query_id].append(doc_id)

# Input u value for MRR@u and MAP@u calculation
u = int(input("Enter the value of u to compute MRR@u and MAP@u: "))

# Calculate precision@k and recall@k for k in {1, 3, 5, 10}
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

# Calculate MAP@u
APs = []
for query_id in top10_predict:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    count_retrieval = 0
    sum_precision = 0
    for i in range(min(u, len(predict_passages))):  # Only consider top u predictions
        if predict_passages[i] in positive_passages:
            count_retrieval += 1
            sum_precision += count_retrieval / (i + 1)
    APs.append(sum_precision / len(positive_passages))

MAP = sum(APs) / len(APs)
print(f"MAP@{u}: ", MAP)

# Calculate MRR@u
MRRs = []
for query_id in top10_predict:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    for i in range(min(u, len(predict_passages))):  # Only consider top u predictions
        if predict_passages[i] in positive_passages:
            MRRs.append(1 / (i + 1))
            break

MRR = sum(MRRs) / len(MRRs)
print(f"MRR@{u}: ", MRR)

# Calculate my_recall divided by the minimum of u and total number of relevant documents
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

# Calculate NDCG@10
ndcgs_10 = []
for query_id in top10_predict:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    dcg = 0.0
    for i in range(min(10, len(predict_passages))):  # Only consider top 10 predictions
        if predict_passages[i] in positive_passages:
            relevance = 1  # Assume relevance is 1 for each relevant document
            dcg += relevance / np.log2(i + 2)  # i+2 because i starts from 0

    # Calculate IDCG@10 (Ideal DCG with relevant documents ranked first)
    ideal_relevances = [1] * min(10, len(positive_passages))  # Relevant documents ideally have relevance 1
    idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances)])

    # Calculate NDCG for the current query
    ndcg = dcg / idcg if idcg > 0 else 0
    ndcgs_10.append(ndcg)

# Calculate the average NDCG@10 across all queries
avg_ndcg_10 = sum(ndcgs_10) / len(ndcgs_10)
print("NDCG@10: ", avg_ndcg_10)
