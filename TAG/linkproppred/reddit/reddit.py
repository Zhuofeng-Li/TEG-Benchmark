import math
import os
import pickle
import shutil
import pandas as pd
import numpy as np
import pickle as pkl
import networkx as nx
import json
import os.path as osp
from typing import List
from huggingface_hub import hf_hub_download

import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from torch_geometric.data import Data
import torch
from torch_geometric.io import fs, read_planetoid_data


class Reddit(InMemoryDataset):
    url = "https://huggingface.co/datasets/ZhuofengLi/TEG-Datasets/blob/main/Reddit/raw"

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'reddit_dataset', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'reddit_dataset', 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        file_names = [
            'reddit_1m.csv',
        ]

        return file_names

	# TODO: add reference code
    def download(self) -> None:
        hf_hub_download(repo_id="ZhuofengLi/TEG-Datasets", filename="reddit_graph.pkl",
                        repo_type="dataset", local_dir=self.raw_dir)

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self) -> None:
        with open(f"{osp.join(self.raw_dir, 'reddit_graph.pkl')}", 'rb') as f:
            data = pickle.load(f)
        self.save([data], self.processed_paths[0])


if __name__ == "__main__":
    dataset = Reddit('.')
    print(dataset[0])
