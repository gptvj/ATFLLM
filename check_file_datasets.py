import datasets

dataset = datasets.load_from_disk("/home/trung/EXPERIMENT_50_HARD_1BATCH_ROUND2_JP/legal_corpus_retrieval_ja")


# for i in range(20):
#     print(len(dataset[i]['positive_passages']))
#     print(len(dataset[i]['negative_passages']))



print(len(dataset))

