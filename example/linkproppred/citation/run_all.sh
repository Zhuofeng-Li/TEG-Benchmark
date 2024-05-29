#!/bin/bash

# Define the models and embedding types
models=("GraphTransformer" "GINE" "EdgeConv" "GraphSAGE" "GeneralConv")
embedding_types=("Angle")

# Loop through models and embedding types
for model in "${models[@]}"; do
    for embedding in "${embedding_types[@]}"; do
        if [ "$model" == "MLP" ]; then
            python mlp.py -et $embedding > link_prediction_log/MLP_${embedding}.txt
        else
            python edge_aware_gnn.py -mt $model -et $embedding > link_prediction_log/${model}_${embedding}.txt
        fi
    done
done

python ../../../utils/find_max.py -fp link_prediction_log > link_prediction_log/summary.txt