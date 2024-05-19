"""
Edge classification with no edge features using the GNNconv layer.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import f1_score
from torch.nn import Linear
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero
from torch_geometric.nn.conv import TransformerConv
import tqdm

from children.goodreads_children import Goodreads_children


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, edge_dim, num_classes):
        super().__init__()
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.book_emb = torch.nn.Embedding(data["book"].num_nodes, hidden_channels)

        self.encoder = GNNEncoder(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels, num_classes)

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict, edge_label_index = data.x_dict, data.edge_index_dict, data.edge_attr_dict, \
            data['user', 'book'].edge_label_index
        x_dict = {
            "user": self.user_emb(data["user"].n_id),
            "book": self.book_emb(data["book"].n_id),
        }
        z_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        return self.decoder(z_dict, edge_label_index)


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_dim):
        super().__init__()
        self.conv1 = TransformerConv((-1, -1), hidden_channels, edge_dim=edge_dim)
        self.conv2 = TransformerConv((-1, -1), out_channels, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super().__init__()
		# TODO: classifier setting is reasonable?
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)  # TODO: update later
        self.lin2 = Linear(hidden_channels, 256)
        self.lin3 = Linear(256, 64)
        self.lin4 = Linear(64, num_classes)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['book'][col]], dim=-1)  # get nodes connected with edge feature
        z = self.lin1(z).relu()
        z = self.lin2(z).relu()
        z = self.lin3(z).relu()
        z = self.lin4(z)
        return z


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


def train(sampled_data):
    model.train()
    pred = model(sampled_data)
    target = sampled_data['user', 'review', 'book'].edge_label
    loss = F.cross_entropy(pred, target)  # TODO: weighted cross entropy
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(sampled_data):
    model.eval()
    pred = model(sampled_data)
    target = sampled_data['user', 'book'].edge_label.float()

    # Calculate Macro-F1
    pred_labels = pred.argmax(dim=1).long().cpu().numpy()
    target_labels = target.round().long().cpu().numpy()
    f1_macro = f1_score(target_labels, pred_labels, average='macro')
    # TODO: other metrics

    return float(f1_macro)


Dataset = Goodreads_children(root='.')
data = Dataset[0]

num_users = data['user'].num_nodes
num_books = data['book'].num_nodes
num_edges = data['user', 'review', 'book'].num_edges

data['user'].x = torch.nn.init.xavier_uniform_(torch.Tensor(num_users, 64))
data['book'].x = torch.nn.init.xavier_uniform_(torch.Tensor(num_books, 64))

npdata = np.load('children_genre/raw/review.npy')
data['user', 'review', 'book'].edge_attr = torch.tensor(npdata).squeeze().float()

del data['book'].y
del data['book'].train_mask
del data['book'].val_mask
del data['book'].test_mask
del data.num_classes

# Add a reverse ('book', 'rev_review', 'user') relation for message passing:
data = T.ToUndirected()(data)
del data['book', 'rev_review', 'user'].edge_label  # Remove "reverse" label.

# Perform a link-level split into training, validation, and test edges:
print(data)
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'review', 'book')],
    rev_edge_types=[('book', 'rev_review', 'user')],
)(data)

edge_label_index = train_data["user", "review", "book"].edge_label_index
edge_label = train_data["user", "review", "book"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    edge_label_index=(("user", "review", "book"), edge_label_index),
    edge_label=edge_label,
    batch_size=1024,
    shuffle=True,
)

edge_label_index = val_data["user", "review", "book"].edge_label_index
edge_label = val_data["user", "review", "book"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(("user", "review", "book"), edge_label_index),
    edge_label=edge_label,
    batch_size=1024,
    shuffle=False,
)

edge_label_index = test_data["user", "review", "book"].edge_label_index
edge_label = test_data["user", "review", "book"].edge_label
test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[20, 10],
    edge_label_index=(("user", "review", "book"), edge_label_index),
    edge_label=edge_label,
    batch_size=1024,
    shuffle=False,
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = Model(hidden_channels=1024, edge_dim=3072, num_classes=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 100):
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data = sampled_data.to(device)
        loss = train(sampled_data)
    # validation
    if epoch % 1 == 0 and epoch != 0:
        print('Validation begins')
        with torch.no_grad():
            preds = []
            ground_truths = []
            f1_macroes = []
            for sampled_data in tqdm.tqdm(test_loader):
                with torch.no_grad():
                    sampled_data = sampled_data.to(device)
                    f1_macro = test(sampled_data)  # TODO: val metric name 
                    f1_macroes.append(f1_macro)
            f1_macroes = torch.tensor(f1_macroes, dtype=torch.float).mean()
            print(f1_macroes)
    # test_rmse = test(test_data)
    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
    #       f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
