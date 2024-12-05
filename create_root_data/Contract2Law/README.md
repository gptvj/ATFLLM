# Step by Step

## Step 1: Chia chunk cho hợp đồng
Bước này thực hiện chia thủ công văn bản hợp đồng thành từng đơn vị chunk sao cho hợp lý và đủ nghĩa
* Đầu vào: raw_contract.docx
* Đầu ra: chunked_contract.json

## Step 2: Tìm kiếm các điều luật liên quan trong bộ corpus ứng với từng chunk hợp đồng: 
find_relevant_laws.ipynb

* Đầu vào: chunked_contract.json và legal_corpus.json
* Đầu ra: find_law_gemini_contract.json

## Step 3: Xử lý format cho dữ liệu ở Step 1
format_data.py

* Đầu vào: find_law_gemini_contract.json
* Đầu ra: find_law_gemini_contract_formatted.json