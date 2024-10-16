## Edge-aware GNN Link Prediction

### MRR Metric
You can utilize following codes to do full batch training on MRR link prediction.
```
python GNN/GNN_Link_MRR.py \
    --use_PLM_node Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_node.pt \
    --use_PLM_edge Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_edge.pt \
    --path Dataset/goodreads_children/LinkPrediction/ \
    --graph_path Dataset/goodreads_children/processed/children.pkl
```

### MRR Metric using neighbor sampling
You can utilize following codes to handle large-scale graph MRR link prediction.

```
python GNN/GNN_Link_MRR_loader.py \
    --use_PLM_node Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_node.pt \
    --use_PLM_edge Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_edge.pt \
    --path Dataset/goodreads_children/LinkPrediction/ \
    --graph_path Dataset/goodreads_children/processed/children.pkl
```

### HitsK Metric
You can utilize following codes to do full batch training on HitsK link prediction.
```
python GNN/GNN_Link_HitsK.py \
    --use_PLM_node Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_node.pt \
    --use_PLM_edge Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_edge.pt \
    --path Dataset/goodreads_children/LinkPrediction/ \
    --graph_path Dataset/goodreads_children/processed/children.pkl
```

### HitsK Metric using neighbor sampling
You can utilize following codes to handle large-scale graph HitsK link prediction.

```
python GNN/GNN_Link_HitsK_loader.py \
    --use_PLM_node Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_node.pt \
    --use_PLM_edge Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_edge.pt \
    --path Dataset/goodreads_children/LinkPrediction/ \
    --graph_path Dataset/goodreads_children/processed/children.pkl
```

### AUC Metric using neighbor sampling
You can utilize following codes to handle large-scale graph AUC link prediction.

```
python GNN/GNN_Link_AUC.py \
    --use_PLM_node Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_node.pt \
    --use_PLM_edge Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_edge.pt \
    --graph_path Dataset/goodreads_children/processed/children.pkl
```

## Edge-aware GNN Node Classification 

```
python GNN/GNN_Node.py \
    --use_PLM_node Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_node.pt \
    --use_PLM_edge Dataset/goodreads_children/emb/children_bert_base_uncased_512_cls_edge.pt \
    --graph_path Dataset/goodreads_children/processed/children.pkl
```

