#!/bin/bash

python GNN/GNN_Node.py \
    --use_PLM_node Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_node.pt \
    --use_PLM_edge Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_edge.pt \
    --graph_path Dataset/goodreads_children/processed/children.pkl

# python GNN/GNN_Node.py \
#     --use_PLM_node Dataset/twitter/emb/twitter_bert_base_uncased_512_cls_node.pt \
#     --use_PLM_edge Dataset/twitter/emb/twitter_bert_base_uncased_512_cls_edge.pt \
#     --graph_path Dataset/twitter/processed/twitter.pkl


