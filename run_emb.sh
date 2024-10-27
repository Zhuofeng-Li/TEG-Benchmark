#!/bin/bash
#!/bin/bash

# # Define an array of dataset configurations
# datasets=(
#     "goodreads_children:children"
#     "goodreads_history:history"
#     "goodreads_comics:comics"
#     "goodreads_crime:crime"
#     # "twitter:twitter"
#     # "amazon_movie:movie"
#     # "amazon_baby:baby"
#     # "reddit:reddit"
# )

# # Loop through each dataset
# for dataset in "${datasets[@]}"; do
#     # Split the dataset string into folder and name
#     IFS=":" read -r folder name <<< "$dataset"
    
#     echo "Processing dataset: $folder"
    
#     python data_preprocess/generate_llm_emb.py \
#         --pkl_file "Dataset/$folder/processed/${name}.pkl" \
#         --path "Dataset/$folder/emb" \
#         --name "$name" \
#         --model_name meta-llama/Meta-Llama-3-8B \
#         --max_length 2048 \
#         --batch_size 25
# done


# python data_preprocess/generate_llm_emb.py \
#     --pkl_file Dataset/arxiv/processed/arxiv.pkl \
#     --path Dataset/arxiv/emb \
#     --name arxiv \
#     --model_name bert-base-uncased \

# python data_preprocess/generate_llm_emb.py \
#     --pkl_file Dataset/arxiv/processed/arxiv.pkl \
#     --path Dataset/arxiv/emb \
#     --name arxiv \
#     --model_name bert-large-uncased \
#     --batch_size 250
 
# python data_preprocess/generate_llm_emb.py \
#     --pkl_file Dataset/twitter/processed/twitter.pkl \
#     --path Dataset/twitter/emb \
#     --name tweets \
#     --model_name bert-base-uncased \

python data_preprocess/generate_llm_emb.py \
    --pkl_file Dataset/twitter/processed/twitter.pkl \
    --path Dataset/twitter/emb \
    --name tweets \
    --model_name bert-base-uncased \

python data_preprocess/generate_llm_emb.py \
    --pkl_file Dataset/twitter/processed/twitter.pkl \
    --path Dataset/twitter/emb \
    --name tweets \
    --model_name bert-large-uncased \
    --batch_size 250

# python data_preprocess/generate_llm_emb.py \
#     --pkl_file Dataset/amazon_movie/processed/movie.pkl \
#     --path Dataset/amazon_movie/emb \
#     --name movie \
#     --model_name bert-base-uncased \


# python data_preprocess/generate_llm_emb.py \
#     --pkl_file Dataset/amazon_baby/processed/baby.pkl \
#     --path Dataset/amazon_baby/emb \
#     --name baby \
#     --model_name bert-base-uncased \

# python data_preprocess/generate_llm_emb.py \
#     --pkl_file Dataset/reddit/processed/reddit.pkl \
#     --path Dataset/reddit/emb \
#     --name reddit \
#     --model_name bert-large-uncased \
