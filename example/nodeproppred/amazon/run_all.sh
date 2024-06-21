#!/bin/bash

# # Define the models, embedding types, and data types
# models=("MLP" "GraphTransformer" "GINE" "EdgeConv" "GraphSAGE" "GeneralConv")
# embedding_types=("Bert")
# data_types=("children" "mystery_thriller_crime" "comics_graphic")  # Add more data types if needed

# Define the models, embedding types, and data types
models=("MLP" "GraphTransformer" "GINE" "EdgeConv" "GraphSAGE" "GeneralConv")
embedding_types=("None")
data_types=("app" "movie")  # Add more data types if needed


# Loop through models, embedding types, and data types
for model in "${models[@]}"; do
    for embedding in "${embedding_types[@]}"; do
        for data_type in "${data_types[@]}"; do
            python edge_aware_gnn.py -mt $model -et $embedding -dt $data_type > ${data_type}/node_classification_log/${model}_${embedding}.txt
        done
    done
done

#python ../../../utils/find_max_node.py -fp mystery_thriller_crime/node_classification_log -fs Large_Bert
# python ../../../utils/find_max_node.py -fp comics_graphic/node_classification_log -fs GPT-3.5
# python ../../../utils/find_max_node.py -fp mystery_thriller_crime/node_classification_log -fs Bert