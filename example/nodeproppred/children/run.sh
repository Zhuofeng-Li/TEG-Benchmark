#!/bin/bash

# # MLP

# # With GPT-3.5-TURBO Embedding:
# python mlp.py -et GPT-3.5-TURBO > node_classification_log/MLP_GPT-3.5-TURBO.txt

# Without Embedding:
# python mlp.py -mt None > node_classification_log/MLP_None.txt

# GraphTransformer

# # With GPT-3.5-TURBO Embedding:
# python edge_aware_gnn.py -mt GraphTransformer -et GPT-3.5-TURBO > node_classification_log/GraphTransformer_GPT-3.5-TURBO.txt

# Without Embedding:
python edge_aware_gnn.py -mt GraphTransformer -et None > node_classification_log/GraphTransformer_None.txt

# GINE

# # With GPT-3.5-TURBO Embedding:
# python edge_aware_gnn.py -mt GINE -et GPT-3.5-TURBO > node_classification_log/GINE_GPT-3.5-TURBO.txt

# Without Embedding:
python edge_aware_gnn.py -mt GINE -et None > node_classification_log/GINE_None.txt

# EdgeConv

# # With GPT-3.5-TURBO Embedding:
# python edge_aware_gnn.py -mt EdgeConv -et GPT-3.5-TURBO > node_classification_log/EdgeConv_GPT-3.5-TURBO.txt

# Without Embedding:
python edge_aware_gnn.py -mt EdgeConv -et None > node_classification_log/EdgeConv_None.txt

# GraphSAGE_mean

# # With GPT-3.5-TURBO Embedding:
# python edge_aware_gnn.py -mt GraphSAGE -et GPT-3.5-TURBO > node_classification_log/GraphSAGE_GPT-3.5-TURBO.txt

# Without Embedding:
python edge_aware_gnn.py -mt GraphSAGE -et None > node_classification_log/GraphSAGE_None.txt

# GeneralConv

# # With GPT-3.5-TURBO Embedding:
# python edge_aware_gnn.py -mt GeneralConv -et GPT-3.5-TURBO > node_classification_log/GeneralConv_GPT-3.5-TURBO.txt

# Without Embedding:
# python edge_aware_gnn.py -mt GeneralConv -et None > node_classification_log/GeneralConv_None.txt
