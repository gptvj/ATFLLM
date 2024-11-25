# Tính precision, recall, map
# Cụ thể tính: precision@k và recall@k với k thuộc {1, 2, 3, 5, 7, 10}
# Tính my_recall thay vì chia cho tổng số tài liệu liên quan thì hãy chia cho giá trị nhỏ nhất của k và tổng số tài liệu liên quan
import datasets
from datasets import load_dataset
import os

all_files = os.listdir("rank_ensemble")
txt_files = [f for f in all_files if f.endswith('.txt')]

list_path = []
for path in txt_files:
    list_path.append("rank_ensemble/{}".format(path))
print(list_path)


for rank_txt_path in list_path:
    dataset_name_or_path = "test_retrieval_ja"

    dataset_name = "/".join(dataset_name_or_path.split('/')[:-1])   
    dataset_language = 'default'
    info = dataset_name_or_path.split('/')
    dataset_split = info[-1] if len(info) == 3 else 'train'
    # print("dataset_split", dataset_split)


    print('==========  {}'.format(rank_txt_path))
    is_hug = False
    # rank_txt_path = "/home/trungquang/Japanese_retrieval_with_LLM_new_propose/ja_validation_embedding/rank_japanese.txt"

    # rank_txt_path = "/home/trungquang/Japanese_retrieval_with_LLM_new_propose/ja_validation_embedding/rank_japanese_model_repllama_round2_checkpoint-30.txt"



    ks = [3, 5, 10, 20, 50, 100, 200]    # Danh sách các k cần tính precision và recall

    # Load lại test dataset
    if is_hug:
        # test_ds = load_dataset(path = dataset_name_or_path,
        #                        name = dataset_language,
        #                     trust_remote_code=True)[dataset_split]
        test_ds = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)

    else: 
        test_ds = datasets.load_from_disk(dataset_name_or_path)

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
    # for query_id in query_relate: 
    #     print("Query id: ", query_id)
    #     print("Query: ", test_ds['query'][count])
    #     print("Positive passages: ", query_relate[query_id])
    #     print("Top 10 predict: ", top10_predict[query_id])
    #     count += 1
    #     if count == 5: 
    #         break

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

    # print("Precisions@k: ", precisions)
    # print("Recalls@k: ", recalls)

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
    # print("MAP@10: ", MAP)

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

    print('\n\n')


# import os
# # print(os.listdir('model_repllama'))
# path_original = []
# for path in os.listdir('model_repllama'):
#     path_original.append('model_repllama/{}'.format(path))

# print(path_original)


# import os



# all_files = os.listdir("ja_validation_embedding")
# txt_files = [f for f in all_files if f.endswith('.txt')]

# all = []
# for path in txt_files:
#     all.append("ja_validation_embedding/{}".format(path))
# print(all)
