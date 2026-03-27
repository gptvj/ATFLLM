import argparse
import json
import os
from datasets import Dataset

def process_corpus(input_file, output_folder, output_name):
    # Load data from the JSON file
    corpus = json.load(open(input_file, encoding='utf-8'))

    # Convert data to the new format
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

    # Create a Hugging Face Dataset from the new data format
    dataset = Dataset.from_dict(data_new)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the dataset to disk
    output_path = os.path.join(output_folder, output_name)
    dataset.save_to_disk(output_path)
    
    print(f"Dataset created successfully at {output_path}")

    # Load the dataset from disk to verify it was saved correctly
    dataset = Dataset.load_from_disk(output_path)
    print(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process legal corpus into custom dataset format.")
    
    # Add command-line arguments
    parser.add_argument('--input_file', type=str, default="vjdatabase/legal_corpus.json", help="Path to the input JSON file.")
    parser.add_argument('--output_folder', type=str, default="japanese_datasets", help="Folder to save the processed dataset.")
    parser.add_argument('--output_name', type=str, default="legal_corpus_retrieval", help="Name of the output dataset.")
    
    args = parser.parse_args()

    # Call the processing function with command-line arguments
    process_corpus(args.input_file, args.output_folder, args.output_name)