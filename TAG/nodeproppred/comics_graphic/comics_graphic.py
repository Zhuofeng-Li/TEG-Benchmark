import json
import os.path as osp
from typing import List

import torch
from torch_geometric.data import InMemoryDataset, HeteroData
from torch_geometric import seed_everything


class Comics(InMemoryDataset):
    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def num_classes(self) -> int:
        assert isinstance(self._data, HeteroData)
        return int(self._data['book'].y.max()) + 1

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'comics_graphic_dataset', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'comics_graphic_dataset', 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        file_names = [
            'node-feat', 'node-label', 'relations', 'split',
            'num-node-dict.csv.gz'
        ]

        return file_names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self) -> None:
        seed_everything(66)

        path = osp.join(self.raw_dir, 'goodreads_reviews_comics_graphic.json')
        genre_path = osp.join(self.raw_dir, 'goodreads_book_genres_initial.json')

        final_data = []
        genres = ['history, historical fiction, biography',
                  'children',
                  'romance',
                  'comics, graphic',
                  'non-fiction',
                  'mystery, thriller, crime',
                  'poetry',
                  'young-adult',
                  'fiction',
                  'fantasy, paranormal']
        bookid2genre = {}

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                final_data.append(data)

        with open(genre_path) as f:
            for line in f:
                data = json.loads(line)
                genre_list = [1 if genre in data['genres'] else 0 for genre in genres]
                bookid2genre[data['book_id']] = genre_list

        user_id2idx = {}
        book_id2idx = {}

        edge_index_user_book = []
        edge_label = []

        multi_label = []

        for item in final_data:
            user_id = item['user_id']
            book_id = item['book_id']

            # user book
            if user_id not in user_id2idx:
                user_id2idx[user_id] = len(user_id2idx)
            if book_id not in book_id2idx:
                book_id2idx[book_id] = len(book_id2idx)
                multi_label.append(bookid2genre[book_id])

            # user - book edge
            edge_index_user_book.append([user_id2idx[user_id], book_id2idx[book_id]])

            # edge label (rating)
            edge_label.append(item['rating'])

        # load to hetordata
        num_users = len(user_id2idx)
        num_books = len(book_id2idx)

        data = HeteroData()
        data['user'].x = torch.nn.init.xavier_uniform_(torch.Tensor(num_users, 64))
        data['book'].x = torch.nn.init.xavier_uniform_(torch.Tensor(num_books, 64))
        data['book'].y = torch.tensor(multi_label).float()
        data['user', 'review', 'book'].edge_index = torch.tensor(edge_index_user_book,
                                                                 dtype=torch.long).t().contiguous()

        data['user', 'review', 'book'].edge_label = torch.tensor(edge_label)

        # data split
        train_ratio = 0.8
        val_ratio = 0.1

        num_book = data['book'].num_nodes
        num_train_book = int(num_book * train_ratio)
        num_val_book = int(num_book * val_ratio)
        num_test_book = num_book - num_train_book - num_val_book

        book_indices = torch.randperm(num_book)

        data['book'].train_mask = torch.zeros(num_book, dtype=torch.bool)
        data['book'].val_mask = torch.zeros(num_book, dtype=torch.bool)
        data['book'].test_mask = torch.zeros(num_book, dtype=torch.bool)

        data['book'].train_mask[book_indices[:num_train_book]] = 1
        data['book'].val_mask[book_indices[num_train_book:num_train_book + num_val_book]] = 1
        data['book'].test_mask[book_indices[-num_test_book:]] = 1

        data.num_classes = 10

        self.save([data], self.processed_paths[0])


if __name__ == '__main__':
    Comic(root='.')
