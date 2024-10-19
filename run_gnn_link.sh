#!/bin/bash

# Define datasets
datasets=(
    # "goodreads_children"
    # "twitter"
    # "goodreads_history"
    "goodreads_crime"
)

# Define models to run
models=(
    "GNN_Link_HitsK_loader.py"
    # "GNN_Link_MRR_loader.py"
    # "GNN_Link_AUC.py"
)

# Loop through datasets and models
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running $model on $dataset dataset"
        python GNN/$model \
            --use_PLM_node Dataset/$dataset/emb/${dataset##*_}_bert_base_uncased_512_cls_node.pt \
            --use_PLM_edge Dataset/$dataset/emb/${dataset##*_}_bert_base_uncased_512_cls_edge.pt \
            --path Dataset/$dataset/LinkPrediction/ \
            --graph_path Dataset/$dataset/processed/${dataset##*_}.pkl \
            --batch_size 1024
    done
done
