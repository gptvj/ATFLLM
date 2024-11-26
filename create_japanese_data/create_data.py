import json
import pickle
import argparse
from datasets import Dataset

# Argument parser
parser = argparse.ArgumentParser(description="Process training data for retrieval tasks.")
parser.add_argument("--train_data", type=str, required=True, help="Path to the training data JSON file.")
parser.add_argument("--corpus_data", type=str, required=True, help="Path to the legal corpus JSON file.")
parser.add_argument("--negative_passages", type=str, required=True, help="Path to the pickle file containing negative passages.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed dataset.")

args = parser.parse_args()

# Load data
train_data = args.train_data
corpus_data = args.corpus_data
negative_passages_file = args.negative_passages
output_dir = args.output_dir

corpus = json.load(open(corpus_data, encoding='utf-8'))
data = json.load(open(train_data, encoding='utf-8'))

print("Before processing train data")
print("Number of items:", len(data['items']))
print("Sample item:", data['items'][0])

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
    return None

# Prepare the new dataset format
data_new = {
    "query_id": [],
    "query": [],
    "positive_passages": [],
    "negative_passages": []
}

# Load negative passages
with open(negative_passages_file, 'rb') as f:
    data_negative = pickle.load(f)
    print("Number of negative passages:", len(data_negative))
    print("Sample negative passage:", data_negative[0])

query_id_count = 1
for item in data['items']:
    data_new['query_id'].append(str(query_id_count))
    query_id_count += 1
    data_new['query'].append(item['question_full'])

    # Add positive passages
    this_pos = []
    for relevant_article in item['relevant_articles']:
        pos_passage = find_in_corpus(relevant_article['law_id'], relevant_article['article_id'])
        if pos_passage:
            this_pos.append(pos_passage)
    data_new['positive_passages'].append(this_pos)

    # Add negative passages
    this_neg = []
    set_id = set([relevant_article['law_id'] + relevant_article['article_id'] for relevant_article in item['relevant_articles']])
    for item_negative in data_negative:
        if item_negative['question'] == item['question_full']:
            doc_id = item_negative['docid']
            article_id = item_negative['article_id']
            if doc_id + article_id not in set_id:
                neg_passage = find_in_corpus(doc_id, article_id)
                if neg_passage:
                    this_neg.append(neg_passage)
    data_new['negative_passages'].append(this_neg)

print("After processing train data")

# Convert to Dataset object and save to disk
dataset = Dataset.from_dict(data_new)
dataset.save_to_disk(output_dir)

# Load and verify the dataset
dataset = Dataset.load_from_disk(output_dir)
print("Loaded dataset:")
print(dataset)

# Print a sample data point
i = 2
print("Query ID:", dataset['query_id'][i])
print("Query:", dataset['query'][i])
print("Positive Passages:", dataset['positive_passages'][i])
print("Negative Passages:", dataset['negative_passages'][i])
print("Number of Negative Passages:", len(dataset['negative_passages'][i]))
