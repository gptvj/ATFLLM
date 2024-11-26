import json
import argparse
from datasets import Dataset

# Argument parser
parser = argparse.ArgumentParser(description="Process validation data for retrieval tasks.")
parser.add_argument("--train_data", type=str, required=True, help="Path to the validation data JSON file.")
parser.add_argument("--corpus_data", type=str, required=True, help="Path to the legal corpus JSON file.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed dataset.")

args = parser.parse_args()

# Load data
train_data = args.train_data
corpus_data = args.corpus_data
output_dir = args.output_dir

corpus = json.load(open(corpus_data, encoding='utf-8'))
data = json.load(open(train_data, encoding='utf-8'))

print("Before processing validation data")
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
    "positive_passages": []
}

# Convert the data
query_id_count = 1
for item in data['items']:
    data_new['query_id'].append(str(query_id_count))
    query_id_count += 1
    data_new['query'].append(item['question_full'])
    this_pos = []
    for relevant_article in item['relevant_articles']:
        pos_passage = find_in_corpus(relevant_article['law_id'], relevant_article['article_id'])
        if pos_passage:
            this_pos.append(pos_passage)
    data_new['positive_passages'].append(this_pos)

print("After processing validation data")

# Convert to Dataset object and save to disk
dataset = Dataset.from_dict(data_new)
dataset.save_to_disk(output_dir)

# Load and verify the dataset
dataset = Dataset.load_from_disk(output_dir)
print("Loaded dataset:")
print(dataset)

# Print a sample data point
print("Query ID:", dataset['query_id'][0])
print("Query:", dataset['query'][0])
print("Positive Passages:", dataset['positive_passages'][0])
