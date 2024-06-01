#!/bin/bash

# # Define the models, embedding types, and data types
# models=("MLP" "GraphTransformer" "GINE" "EdgeConv" "GraphSAGE" "GeneralConv")
# embedding_types=("Bert")
# data_types=("children" "mystery_thriller_crime" "comics_graphic")  # Add more data types if needed

# Define the models, embedding types, and data types
models=("MLP" "GraphTransformer" "GINE" "EdgeConv" "GraphSAGE" "GeneralConv")
embedding_types=("Large_Bert")
data_types=("mystery_thriller_crime")  # Add more data types if needed


# Loop through models, embedding types, and data types
for model in "${models[@]}"; do
    for embedding in "${embedding_types[@]}"; do
        for data_type in "${data_types[@]}"; do
            python edge_aware_gnn.py -mt $model -et $embedding -dt $data_type > ${data_type}/link_prediction_log/${model}_${embedding}.txt
        done
    done
done

python ../../../utils/find_max_link.py -fp mystery_thriller_crime/link_prediction_log -fs Large_Bert 
# python ../../../utils/find_max_link.py -fp mystery_thriller_crime/link_prediction_log -fs Bert 
# python ../../../utils/find_max_link.py -fp comics_graphic/link_prediction_log -fs Bert 