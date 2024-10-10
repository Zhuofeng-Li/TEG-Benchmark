#!/bin/bash

python data_preprocess/generate_llm_emb.py \
    --pkl_file Dataset/goodreads_history/processed/history.pkl \
    --path Dataset/history/emb \
    --name history \
    --model_name bert-base-uncased \