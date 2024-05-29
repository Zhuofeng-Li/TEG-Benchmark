#!/bin/bash

python mlp.py -mt None > node_classification_log/MLP_None.txt

python edge_aware_gnn.py -mt GraphTransformer -et None > node_classification_log/GraphTransformer_None.txt

python edge_aware_gnn.py -mt GINE -et None > node_classification_log/GINE_None.txt

python edge_aware_gnn.py -mt EdgeConv -et None > node_classification_log/EdgeConv_None.txt

python edge_aware_gnn.py -mt GraphSAGE -et None > node_classification_log/GraphSAGE_None.txt

python edge_aware_gnn.py -mt GeneralConv -et None > node_classification_log/GeneralConv_None.txt
