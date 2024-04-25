# Goodreads children

## Requirements

+ **PyG >= 2.4**

## Introduction

+ We only select `user` and `book` as nodes, `review` as edge, `book genre` as multi-label
+ Our task is multi-label node classification

## Data

+ `x`: [num_nodes, num_feature] (init by xavier_uniform_)
+ `y`: [num_nodes, num_classes]
+ `edge_index`: [2, num_edges]
+ `edge_attrs`: [num_edges, num_text_feature] (from edge-text embeddings)

## Set up

+ Create dir as follows:

```
  ├─children
  |    ├─goodreads_children.py
  |    ├─__init__.py
  |    ├─raw
  |    |  ├─goodreads_book_genres_initial.json
  |    |  └goodreads_reviews_children.json
```

## Downstream Tasks

### Multi-label classification

```
cd downstream_tasks/
python multi_label_classification.py
```

## Reference

Please read following materials carefully to set up your own dataset!

+ [pyg graph dataset](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html)
+ [ogbn-mag HeteroData example](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/ogb_mag.html)
+ [Heterogeneous Graph Learning](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html)