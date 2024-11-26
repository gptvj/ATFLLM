# BM25plus + Ours (ckpt*)
# bm25 score
# alpha = 0.001
# # best round2 -400
# beta = 0.6
# # best round 2 - 1600
# gamma = 0.4
# ------------- BEST CONFIG IN BM25+ + CKPT 400 ROUND2 + CKPT 1600 ROUND2

import pandas as pd

def merge_and_multiply_scores(llama_file, bm25_file, llama2_file, output_file, alpha, beta, gamma):
    # Đọc file llama_rank.txt
    llama_data = pd.read_csv(llama_file, sep="\t", header=None, names=["query_id", "corpus_id", "llama_score"])

    # Đọc file bm25_rank.txt
    bm25_data = pd.read_csv(bm25_file, sep="\t", header=None, names=["query_id", "corpus_id", "bm25_score"])

    # Đọc file llama2_rank.txt
    llama2_data = pd.read_csv(llama2_file, sep="\t", header=None, names=["query_id", "corpus_id", "llama2_score"])



    # Gộp ba bảng dựa trên cột query_id và corpus_id
    merged_data = pd.merge(bm25_data, llama_data, on=["query_id", "corpus_id"], how='inner')
    merged_data = pd.merge(merged_data, llama2_data, on=["query_id", "corpus_id"], how='inner')
    merged_data = merged_data.drop_duplicates()

    # Tạo cột score_merge_multi bằng cách kết hợp bm25_score, llama_score và llama2_score
    merged_data["score_merge_multi"] = alpha * merged_data["bm25_score"] + beta * merged_data["llama_score"] + gamma * merged_data["llama2_score"]


    # Sắp xếp theo query_id tăng dần và score_merge_multi giảm dần
    merged_data = merged_data.sort_values(by=["query_id", "score_merge_multi"], ascending=[True, False])

    # Chọn chỉ cột query_id, corpus_id và score_merge_multi
    final_data = merged_data[["query_id", "corpus_id", "score_merge_multi"]]

    # Lưu vào file kết quả
    final_data.to_csv(output_file, sep="\t", index=False, header=False)

    print(f"File kết hợp và nhân điểm đã được lưu tại {output_file}")

# Đường dẫn tới các file llama_rank.txt, bm25_rank.txt và llama2_rank.txt
bm25_rank_txt_path = "temp/sorted_bm25_rank.txt"
llama_rank_txt_path = "temp/jp_full_cp_ckpt_round2_400/rank_japanese.txt"
llama2_rank_txt_path = "temp/jp_round1_2800/rank_japanese.txt"
output_file = "temp/rank_ensemble/merge_rank.txt"

import os
if not os.path.exists(os.path.dirname(output_file)): 
    os.makedirs(os.path.dirname(output_file))

# bm25 score
alpha = 0.001
# best round2 -400
beta = 0.6
# best round 1 - 2800
gamma = 0.4

# Gọi hàm để gộp và nhân điểm
merge_and_multiply_scores(llama_rank_txt_path, bm25_rank_txt_path, llama2_rank_txt_path, output_file, alpha, beta, gamma)
