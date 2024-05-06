# Goodreads children

## Requirements

+ **PyG >= 2.4**

## Introduction

+ We only select `user` and `book` as nodes, `review` as the edge, and `book genre` as the multi-label.
+ Our tasks are multi-label node classification, link prediction, and edge classification.

## Data

+ `x`: [num_nodes, num_feature] (init by xavier_uniform_)
+ `y`: [num_nodes, num_classes]
+ `edge_index`: [2, num_edges]
+ `edge_attrs`: [num_edges, num_text_feature] (from edge-text embeddings)
+ `edge_label`: [num_edges] (review rating)

## Set up

+ Get data:

``` 
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz -O goodreads_reviews_children.json.gz
gzip -d goodreads_reviews_children.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_genres_initial.json.gz -O goodreads_book_genres_initial.json.gz
gzip -d goodreads_book_genres_initial.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books_children.json.gz -O goodreads_books_children.json.gz
gzip -d goodreads_books_children.json.gz
```

+ we provide tow types of goodreads_children datasets: (1) `children` dataset consists of user-review-book. (2) `children_genre` dataset consists of user-review-book, book-description-genre. Set up the dataset as follows:

```
  ├─children
  |    ├─goodreads_children.py
  |    ├─__init__.py
  |    ├─raw
  |    |  ├─goodreads_book_genres_initial.json
  |    |  └goodreads_reviews_children.json

```

```
├─children_genre
|       ├─goodreads_children_genre.py
        |─__init__.py
|       ├─raw
|       |  ├─goodreads_books_children.json
|       |  ├─goodreads_book_genres_initial.json
|       |  └goodreads_reviews_children.json
```

## Downstream Tasks

Please refer [Edgeformer](https://openreview.net/pdf?id=2YQrqe4RNv) for more experimental details (we follow its settings completely).

### Multi-label classification (children dataset)

```
cd downstream_tasks/
python multi_label_classification.py
```

### Link prediction (children_genre dataset)

```
cd downstream_tasks/
python link_prediction.py
```

## Reference

Please read the following materials carefully to set up your dataset!

+ [pyg graph dataset](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html)
+ [ogbn-mag HeteroData example](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/ogb_mag.html)
+ [Heterogeneous Graph Learning](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html)
+ [Link Prediction on Heterogeneous Graphs with PyG](https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70)
