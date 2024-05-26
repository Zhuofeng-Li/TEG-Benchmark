# Goodreads children

## Requirements

+ **PyG >= 2.4**

## Project format 
+ `example/`: baseline files for the downstream tasks, e.g., node classification and link prediction.
+ `TAG/`: the pyg dataset preprocess files.

**I have set up the Goodreads_children dataset. Please follow my format for other datasets!**
``` 
├─example
│  ├─linkproppred
│  │  └─children
│  │      └─children_dataset
│  └─nodeproppred
│      └─children
└─TAG
    ├─linkproppred
    │  └─children
    ├─nodeproppred
    │  └─children
```

## Children Data format 

+ `x`: [num_nodes, num_feature] (init by xavier_uniform_)
+ `y`: [num_nodes, num_classes]
+ `edge_index`: [2, num_edges]
+ `edge_attrs`: [num_edges, num_text_feature] (from edge-text embeddings)
+ `edge_label`: [num_edges] (review rating)

## Reference

Please read the following materials carefully to set up your dataset!

+ [pyg graph dataset](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html)
+ [ogbn-mag HeteroData example](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/ogb_mag.html)
+ [Heterogeneous Graph Learning](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html)
+ [Link Prediction on Heterogeneous Graphs with PyG](https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70)
