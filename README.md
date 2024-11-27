<div align="center">
   <img src="./Images/logo-vj.c7683b6.png" width="800" height="200">
</div>
<div align="center">
   <img src="./Images/pipeline.jpg" alt="Pipeline Image" width="1020" height="600">
</div>

# Legal Document Retrieval - Japanese Dataset

- This repository provides the workflow and scripts for training and evaluating a legal document retrieval system using BM25 and fine-tuned models. The workflow involves multiple rounds of training, the creation of hard negatives, and ensemble evaluation.
- Tevatron is a library for **dense retrieval** tasks. We inherited the code from Tevatron, custom-implemented the necessary modules, and performed model quantization prior to training due to hardware limitations. You can access the full source code at the following GitHub repository:
[Tevatron GitHub Repository](https://github.com/texttron/tevatron)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Dataset Creation](#dataset-creation)
4. [Training and Evaluation Workflow](#training-and-evaluation-workflow)
5. [Ensemble Evaluation](#ensemble-evaluation)

---

## Introduction

This project is focused on fine-tuning language models for legal document retrieval in Japanese. It includes:
- Creating a Japanese dataset from root data.
- Iterative training rounds with hard negatives.
- Ensemble techniques combining BM25 and fine-tuned models for improved performance.

---

## Requirements

Ensure the following are installed:
- Python 3.9.19
   ```bash
   conda create -n new_atf python==3.9.19
   ```
- Required libraries from `requirements.txt`
   ```bash
   pip install -r requirements.txt
   ```
- You can run this script for install all: 
[install_lib.sh](./install_lib.sh)
   ```bash
   cd tevatron
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   cd ..
   ```
---

## Dataset Creation

### Step 1: Create Japanese Dataset
Use the `create_japanese_data` directory to prepare the dataset.

1. **Download root data from huggingface**  
python load_data_from_hub.py

2. **Generate the corpus, training and test datasets**  
You can see more details in the [scripts.md](./create_japanese_data/scripts.md) file.

---

## Training and Evaluation Workflow

### Step 2: Training - Round 1
Run the following script to start the first training round:
   ```bash
   ./run_train_round1.sh
   ```

### Step 3: Create Hard Negatives for Round 2

1. **Generate ranked files using the model trained in round 1:**  
   ```bash
   create_hardnegative_data/create_top_negative_round2.sh
   ```
2. **Resolve NaN errors and replace with BM25 negatives**  
   ```bash
   python create_hardnegative_data/create_hard_negative_resolve_nan_error.py
   ```

### Step 4: Training - Round 2
Run the following script to start the first training round:
   ```bash
   ./run_train_round2.sh
   ```

### Step 5: Evaluate

1. **Evaluate Round 1 Model**  
   ```bash
   python evaluate_metrics.py --test_ds_path japanese_datasets/test_retrieval_ja --rank_txt_path temp/jp_round1_2800/rank_japanese.txt
   ```
2. **Evaluate Round 2 Model**  
   ```bash
   python evaluate_metrics.py --test_ds_path japanese_datasets/test_retrieval_ja --rank_txt_path temp/jp_full_cp_ckpt_round2_400/rank_japanese.txt
   ```

---
## Ensemble Evaluation

### Step 6: Ensemble Rankings

1. **Create BM25 rank file**  
   ```bash
   python ensemble/create_rank_bm25.py
   ```

2. **Evaluate Round 2 Model**  
   ```bash
   python ensemble/ensemble_bm25_llm1_llm2.py
   ```

2. **Evaluate the BM25 ensemble**  
Evaluate the BM25 ensemble:
   ```bash
   python evaluate_metrics.py --test_ds_path japanese_datasets/test_retrieval_ja --rank_txt_path temp/sorted_bm25_rank.txt
   ```

Evaluate the merged ensemble:
   ```bash
   python evaluate_metrics.py --test_ds_path japanese_datasets/test_retrieval_ja --rank_txt_path temp/rank_ensemble/merge_rank.txt
   ```

---

### Results
Include a summary of metrics such as recall, precision, MAP, MRR, NDCG.

```bash
Precision and Recall at different k values:
+-----+-------------+----------+
|  k  | Precision@k | Recall@k |
+-----+-------------+----------+
|  3  |    0.4128   |  0.6437  |
|  5  |    0.3108   |  0.7632  |
|  10 |    0.1769   |  0.8369  |
|  20 |    0.0958   |  0.8864  |
|  50 |    0.0398   |  0.9213  |
| 100 |    0.0210   |  0.9592  |
| 200 |    0.0109   |  0.9841  |
+-----+-------------+----------+

MAP and MRR:
+--------+--------+
| Metric | Value  |
+--------+--------+
| MAP@10 | 0.6667 |
| MRR@10 | 0.8420 |
+--------+--------+

My Recall at different k values:
+-----+-------------+
|  k  | My_recall@k |
+-----+-------------+
|  3  |    0.6846   |
|  5  |    0.7700   |
|  10 |    0.8373   |
|  20 |    0.8864   |
|  50 |    0.9213   |
| 100 |    0.9592   |
| 200 |    0.9841   |
+-----+-------------+

NDCG@10: 0.7472
```

---

### Citation
If you use this work in your research, please cite it as:

*Update later*
