#!/bin/bash

# Define the models and embedding types
# models=("MLP" "GraphTransformer" "GINE" "EdgeConv" "GraphSAGE" "GeneralConv")
# embedding_types=("GPT-3.5-TURBO" "Large_Bert" "None")
# data_types=("citation")  # Add more data types if needed

models=("MLP" "GraphTransformer" "GINE" "EdgeConv" "GraphSAGE" "GeneralConv")
embedding_types=("None")
data_types=("citation")  # Add more data types if needed

# Loop through models, embedding types, and data types
for model in "${models[@]}"; do
    for embedding in "${embedding_types[@]}"; do
        for data_type in "${data_types[@]}"; do
            python edge_aware_gnn.py -mt $model -et $embedding -dt $data_type > node_classification_log/${model}_${embedding}.txt
        done
    done
done

python ../../../utils/find_max_multi_class_node.py -fp node_classification_log > node_classification_log/summary.txt -fs None

