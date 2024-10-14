#!/bin/bash

python data_preprocess/generate_llm_emb.py \
    --pkl_file Dataset/goodreads_history/processed/history.pkl \
    --path Dataset/goodreads_history/emb \
    --name history \
    --model_name bert-base-uncased \
 
# python data_preprocess/generate_llm_emb.py \
#     --pkl_file Dataset/twitter/processed/twitter.pkl \
#     --path Dataset/twitter/emb \
#     --name tweets \
#     --model_name bert-base-uncased \

# python data_preprocess/generate_llm_emb.py \
#     --pkl_file Dataset/amazon_movie/processed/movie.pkl \
#     --path Dataset/amazon_movie/emb \
#     --name movie \
#     --model_name bert-base-uncased \