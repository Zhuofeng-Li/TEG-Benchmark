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
+ Get data:
```
!wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz -O goodreads_reviews_children.json.gz
!gzip -d goodreads_reviews_children.json.gz
!wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_genres_initial.json.gz -O goodreads_book_genres_initial.json.gz
!gzip -d goodreads_book_genres_initial.json.gz
```
+ Create dir as follows:

├── children

│  ├── raw

│  │  ├── goodreads_reviews_children.json

│  │  ├── goodreads_book_genres_initial

## Reference

Please read the following materials carefully to set up your dataset!

+ [pyg graph dataset](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html)
+ [ogbn-mag HeteroData example](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/ogb_mag.html)
+ [Heterogeneous Graph Learning](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html)
