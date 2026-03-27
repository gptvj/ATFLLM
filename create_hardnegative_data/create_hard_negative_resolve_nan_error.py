import argparse
import datasets

# Setup argument parser
parser = argparse.ArgumentParser(description='Create a hard negative dataset for training.')

# Add arguments for paths
parser.add_argument('--test_ds_path', type=str, default="japanese_datasets/train_retrieval_ja", help='Path to the test dataset.')
parser.add_argument('--legal_ds_path', type=str, default="japanese_datasets/legal_corpus_retrieval_ja", help='Path to the legal dataset.')
parser.add_argument('--rank_txt_path', type=str, default="temp/japanese_embedding_negative_round2/rank_japanese.txt", help='Path to the ranking file.')
parser.add_argument('--save_path', type=str, default="hard_negative_ja_round1_for_training_round2", help='Path to save the hard negative dataset.')
parser.add_argument('--n', type=int, default=50, help='Number of hard negatives per query.')

# Parse the arguments
args = parser.parse_args()

# init
test_ds_path = args.test_ds_path
legal_ds_path = args.legal_ds_path
rank_txt_path = args.rank_txt_path
save_path = args.save_path
n = args.n   # number of hard negative passages per query

# Load the test dataset and legal dataset
test_ds = datasets.load_from_disk(test_ds_path)
legal_ds = datasets.load_from_disk(legal_ds_path)

# Convert legal_ds to a dictionary for faster lookup
legal_dict = {doc_id: {"docid": doc_id, "title": title, "text": text}
              for doc_id, title, text in zip(legal_ds['docid'], legal_ds['title'], legal_ds['text'])}

# Load top100_predict from the file
top100_predict = {}
is_NaN_score = {}
with open(rank_txt_path, "r", encoding='utf-8') as f: 
    for line in f:
        query_id, doc_id, score = line.strip().split("\t")[:3]
        score = float(score)
        top100_predict.setdefault(query_id, []).append(doc_id)
        if query_id not in is_NaN_score:
            is_NaN_score[query_id] = True if score < 0 else False

# Create a new dataset for hard negatives
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
    
    # Add hard negative passages
    list_pos_id = {pos['docid'] for pos in positive_passages}
    nega_this = []
    if is_NaN_score[query_id]: 
        # Get negative passages from the test dataset
        print("NaN")
        count_NaN += 1
        nega_this = [nega for nega in test_ds['negative_passages'][i]][:n]
    else: 
        nega_this = [legal_dict[doc_id] for doc_id in predict_passages if doc_id not in list_pos_id][:n]
    

    new_hard_negative_dataset['negative_passages'].append(nega_this)

print("count_NaN", count_NaN)

# Save the new hard negative dataset to disk
dataset = datasets.Dataset.from_dict(new_hard_negative_dataset)
dataset.save_to_disk(save_path)

# Load the dataset from disk to verify it was saved correctly
dataset = datasets.load_from_disk(save_path)
print(dataset)
print("Done")
