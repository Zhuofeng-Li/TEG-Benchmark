#!/bin/bash

# Define the models and embedding types
models=("MLP" "GraphTransformer" "GINE" "EdgeConv" "GraphSAGE" "GeneralConv")
embedding_types=("Angle" "None")

# Loop through models and embedding types
for model in "${models[@]}"; do
    for embedding in "${embedding_types[@]}"; do
        if [ "$model" == "MLP" ]; then
            python mlp.py -et $embedding > node_classification_log/MLP_${embedding}.txt
        else
            python edge_aware_gnn.py -mt $model -et $embedding > node_classification_log/${model}_${embedding}.txt
        fi
    done
done

python ../../../utils/find_max.py -fp node_classification_log > node_classification_log/summary.txt