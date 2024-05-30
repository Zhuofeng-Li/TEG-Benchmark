import json
import os.path as osp
from typing import List

import torch
from torch_geometric.data import InMemoryDataset, HeteroData
from tqdm import tqdm


class Comics(InMemoryDataset):
    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.load(self.processed_paths[0], data_cls=HeteroData)

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
        path = osp.join(self.raw_dir, 'goodreads_reviews_comics_graphic.json')
        genre_path = osp.join(self.raw_dir, 'goodreads_book_genres_initial.json')

        final_data = []
        final_genre_book = {}
        genres = {'history, historical fiction, biography': 0,
                  'children': 1,
                  'romance': 2,
                  'comics, graphic': 3,
                  'non-fiction': 4,
                  'mystery, thriller, crime': 5,
                  'poetry': 6,
                  'young-adult': 7,
                  'fiction': 8,
                  'fantasy, paranormal': 9,
                  'None': 10}

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                final_data.append(data)

        with open(genre_path) as f:
            for line in f:
                data = json.loads(line)
                main_genre = max(data['genres'], key=data['genres'].get) if data['genres'] else 'None'
                final_genre_book[data['book_id']] = genres[main_genre]

        user_id2idx = {}
        book_id2idx = {}
        edge_index_user_book = []
        edge_index_book_genre = []
        edge_label = []

        for item in tqdm(final_data):
            user_id = item['user_id']
            book_id = item['book_id']

            # user book
            if user_id not in user_id2idx:
                user_id2idx[user_id] = len(user_id2idx)
            if book_id not in book_id2idx:
                book_id2idx[book_id] = len(book_id2idx)

            # user-review-book edge, book-description-genre edge
            edge_index_user_book.append([user_id2idx[user_id], book_id2idx[book_id]])

            # edge label (rating)
            edge_label.append(item['rating'])

        for book_id, genre in tqdm(final_genre_book.items()):
            if book_id in book_id2idx:
                edge_index_book_genre.append([book_id2idx[book_id], genre])

        # load to heterodata
        num_users = len(user_id2idx)
        num_books = len(book_id2idx)

        data = HeteroData()
        data['user'].x = torch.nn.init.xavier_uniform_(torch.Tensor(num_users, 64))
        data['book'].x = torch.nn.init.xavier_uniform_(torch.Tensor(num_books, 64))
        data['genre'].x = torch.nn.init.xavier_uniform_(torch.Tensor(len(genres), 64))

        data['user', 'review', 'book'].edge_index = torch.tensor(edge_index_user_book,
                                                                 dtype=torch.long).t().contiguous()
        data['book', 'description', 'genre'].edge_index = torch.tensor(edge_index_book_genre,
                                                                       dtype=torch.long).t().contiguous()
        data['user', 'review', 'book'].edge_label = torch.tensor(edge_label)

        self.save([data], self.processed_paths[0])


