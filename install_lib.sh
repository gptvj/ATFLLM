#!/bin/bash

# Cài đặt các thư viện cần thiết
pip install datasets
pip install rank_bm25
pip install fugashi
pip install mecab-python3
pip install unidic-lite
pip3 install torch torchvision torchaudio
pip install transformers==4.42.4
pip install peft
pip install bitsandbytes
pip install xformers
pip install faiss-gpu==1.7.2
pip install prettytable

# Thêm thư mục tevatron vào PYTHONPATH
cd tevatron
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ..