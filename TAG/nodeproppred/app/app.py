import json
import os.path as osp
from typing import List
import gzip
import shutil
import torch
from torch_geometric.data import InMemoryDataset, HeteroData
from torch_geometric import seed_everything

class Amazon_Apps(InMemoryDataset):
    def __init__(self, root: str) -> None:
        super().__init__(root)
        print(f'Raw dir: {self.raw_dir}')
        print(f'Processed dir: {self.processed_dir}')
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root,'app_dataset','raw') 

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'app_dataset', 'processed')

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
        
        path = osp.join(self.raw_dir, 'reviews_Apps_for_Android_5.json')
        genre_path = osp.join(self.raw_dir, 'meta_Apps_for_Android.json')
        # 添加打印语句
        print(f'Review path: {path}')
        print(f'Genre path: {genre_path}')

        final_data = []
        categories = ['Kids', 'Music', 'Music Players', 'Reference', 'Games', 'Productivity', 'Navigation', 'Entertainment',
                      'Novelty', 'Books & Comics', 'Books & Readers', 'Radio', 'Health & Fitness', 'Utilities', 'Education',
                      'Lifestyle', 'Calculators', 'Battery Savers', 'Finance', 'Photography', 'Social Networking', 'Shopping',
                      'News & Magazines', 'News', 'Weather', 'Newspapers', 'Artists', 'Alarms & Clocks', 'Video Games', 'Fire TV',
                      'Digital Games', 'Real Estate', 'Calendars', 'Themes', 'Cooking', 'Communication', 'Instruments', 'Podcasts',
                      'Travel', 'Sports', 'Ringtones', 'Comic Strips', 'Web Browsers', "Children's Books", 'City Info', 'Songbooks',
                      'Graphic Novels', 'Notes', 'Magazines', 'Kindle Store', 'Kindle Newspapers', 'North America', 'U.S.',
                      'Kindle Magazines', 'Lifestyle & Culture', 'Arts & Entertainment', 'Regional & Travel', 'Internet & Technology',
                      'Manga', 'Science', 'News, Politics & Opinion','None'] #共62类

        appid2category = {}
        print(path)
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                final_data.append(data)
        with open(genre_path) as f:
            for line in f:
                data = json.loads(line)
                data_cate = [y for x in data['categories'] for y in x]
                genre_list = [1 if category in data_cate else 0 for category in categories]
                appid2category[data['asin']] = genre_list
        

        reviewer_id2idx = {}
        app_id2idx = {}

        edge_index_reviewer_app = []
        edge_label = []

        multi_label = []
        for item in final_data:
            reviewer_id = item['reviewerID']
            app_id = item['asin']

            # user book
            if reviewer_id not in reviewer_id2idx:
                reviewer_id2idx[reviewer_id] = len(reviewer_id2idx)
            if app_id not in app_id2idx:
                app_id2idx[app_id] = len(app_id2idx)

            # user - book edge
            edge_index_reviewer_app.append([reviewer_id2idx[reviewer_id], app_id2idx[app_id]])
            # book label
            multi_label.append(appid2category[app_id])
            # edge label (rating)
            edge_label.append(item['overall'])

        # load to hetordata
        num_users = len(reviewer_id2idx)
        num_books = len(app_id2idx)

        data = HeteroData()
        data['user'].x = torch.nn.init.xavier_uniform_(torch.Tensor(num_users, 64))  # TODO
        data['book'].x = torch.nn.init.xavier_uniform_(torch.Tensor(num_books, 64))
        data['book'].y = torch.tensor(multi_label).float()
        data['user', 'review', 'book'].edge_index = torch.tensor(edge_index_reviewer_app,dtype=torch.long).t().contiguous()

        data['user', 'review', 'book'].edge_label = torch.tensor(edge_label)

        # data split
        train_ratio = 0.8
        val_ratio = 0.1

        num_app = data['book'].num_nodes
        num_train_app = int(num_app * train_ratio)
        num_val_app = int(num_app * val_ratio)
        num_test_app = num_app - num_train_app - num_val_app

        app_indices = torch.randperm(num_app)

        data['book'].train_mask = torch.zeros(num_app, dtype=torch.bool)
        data['book'].val_mask = torch.zeros(num_app, dtype=torch.bool)
        data['book'].test_mask = torch.zeros(num_app, dtype=torch.bool)

        data['book'].train_mask[app_indices[:num_train_app]] = 1
        data['book'].val_mask[app_indices[num_train_app:num_train_app + num_val_app]] = 1
        data['book'].test_mask[app_indices[-num_test_app:]] = 1

        data.num_classes = 62

        self.save([data], self.processed_paths[0])


if __name__ == '__main__':
    Amazon_Apps(root='.')
