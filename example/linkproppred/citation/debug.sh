#!/bin/bash

## MLP 

# # With Angle Embedding:
python mlp.py -et Angle > link_prediction_log/MLP_Angle.txt

# # Without Embedding:
python mlp.py -mt None > link_prediction_log/MLP_None.txt

## GraphTransformer

# With Angle Embedding:
python edge_aware_gnn.py -mt GraphTransformer -et Angle > link_prediction_log/GraphTransformer_Angle.txt

# Without Embedding:
python edge_aware_gnn.py -mt GraphTransformer -et None > link_prediction_log/GraphTransformer_None.txt

## GINE

# With Angle Embedding:
python edge_aware_gnn.py -mt GINE -et Angle > link_prediction_log/GINE_Angle.txt

# Without Embedding:
python edge_aware_gnn.py -mt GINE -et None > link_prediction_log/GINE_None.txt

## EdgeConv

# With Angle Embedding:
python edge_aware_gnn.py -mt EdgeConv -et Angle > link_prediction_log/EdgeConv_Angle.txt

# Without Embedding:
python edge_aware_gnn.py -mt EdgeConv -et None > link_prediction_log/EdgeConv_None.txt

## GraphSAGE_mean

# With Angle Embedding:
python edge_aware_gnn.py -mt GraphSAGE -et Angle > link_prediction_log/GraphSAGE_Angle.txt

# Without Embedding:
python edge_aware_gnn.py -mt GraphSAGE -et None > link_prediction_log/GraphSAGE_None.txt

## GeneralConv

# With Angle Embedding:
python edge_aware_gnn.py -mt GeneralConv -et Angle > link_prediction_log/GeneralConv_Angle.txt

# Without Embedding:
python edge_aware_gnn.py -mt GeneralConv -et None > link_prediction_log/GeneralConv_None.txt