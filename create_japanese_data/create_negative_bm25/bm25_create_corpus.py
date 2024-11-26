import json
import os
import argparse
from tqdm import tqdm


def load_json(corpus_path):
    """Load the JSON data from a given file path."""
    with open(corpus_path, encoding='utf-8') as file:
        data = json.load(file)
    return data["items"]


if __name__ == '__main__':

    # Thiết lập tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Script to process legal corpus data")
    parser.add_argument("--data_dir", default="vjdatabase", type=str, 
                        help="Path to the directory containing the corpus data")
    parser.add_argument("--corpus_file", default="legal_corpus.json", type=str, 
                        help="Name of the corpus file")
    parser.add_argument("--save_dir", default="temp/generated_data", type=str, 
                        help="Directory where the processed data will be saved")
    parser.add_argument("--save_file", default="legal_dict.json", type=str, 
                        help="File where the processed data will be saved")
    
    # Parse các tham số
    args = parser.parse_args()

    # Tạo thư mục lưu trữ kết quả nếu chưa tồn tại
    os.makedirs(args.save_dir, exist_ok=True)

    # Đường dẫn đến file corpus
    corpus_path = os.path.join(args.data_dir, args.corpus_file)

    # Đọc dữ liệu từ corpus
    with open(corpus_path, encoding='utf-8') as file:
        data = json.load(file)

    save_dict = {}

    # Duyệt qua các bài viết pháp lý và trích xuất thông tin
    count = 0
    for law_article in tqdm(data):
        law_id = law_article["law_id"]
        law_articles = law_article["articles"]

        for sub_article in law_articles:
            article_id = str(sub_article["article_id"])
            article_title = sub_article["title"]
            article_text = sub_article["text"]

            # Tạo khóa concat_id duy nhất cho mỗi bài viết
            concat_id = f"{law_id}_{article_id}"
            
            # Nếu concat_id chưa có trong từ điển thì thêm mới
            if concat_id not in save_dict:
                count += 1
                save_dict[concat_id] = {
                    "title": article_title, "text": article_text}

    print(f"Number of unique articles: {count}")
    print("Creating legal dict from raw data...")

    # Lưu kết quả vào file JSON
    output_file = os.path.join(args.save_dir, args.save_file)
    with open(output_file, "w", encoding='utf-8') as outfile:
        json.dump(save_dict, outfile, ensure_ascii=False, indent=4)

    print(f"Finished creating the legal dict. Data saved to {output_file}.")
