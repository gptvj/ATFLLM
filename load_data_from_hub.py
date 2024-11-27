from huggingface_hub import hf_hub_download
import os

# Thay "gptvj/vjdatabase" bằng tên dataset của bạn
REPO_ID = "gptvj/vjdatabase"
FILE_NAMES = [
    "legal_corpus.json",
    "test_retrieval_data.json",
    "train_retrieval_data.json",
    "validation_retrieval_data.json"
]

# Đường dẫn thư mục đích
LOCAL_DIR = "./vjdatabase"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(LOCAL_DIR, exist_ok=True)

# Vòng lặp tải tất cả file vào thư mục đích
for file_name in FILE_NAMES:
    file_path = hf_hub_download(
        repo_id=REPO_ID, 
        filename=file_name, 
        repo_type="dataset", 
        local_dir=LOCAL_DIR
    )
    print(f"Downloaded {file_name} to {file_path}")
