import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from models import MLP, EdgeConvConv
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import global_add_pool, to_hetero
from torch.nn import Linear
from sklearn.metrics import f1_score
import argparse

from TAG.linkproppred.children import Children


class EdgeConvNet(torch.nn.Module):
    def __init__(self, in_channels_nodes, in_channels_edges, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.lin_init_node = MLP(in_channels_nodes, hidden_channels, hidden_channels, num_layers=2)
        self.lin_init_edge = MLP(in_channels_edges, hidden_channels, hidden_channels, num_layers=2)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            nn = MLP(hidden_channels * 3, hidden_channels, hidden_channels, num_layers=2)
            self.convs.append(EdgeConvConv(nn=nn, eps=0.1))

        nn = MLP(hidden_channels * 3, hidden_channels, out_channels, num_layers=2)
        self.convs.append(EdgeConvConv(nn=nn, eps=0.1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.lin_init_node(x)
        edge_attr = self.lin_init_edge(edge_attr)
        for conv in self.convs[:-1]:
            x = x + conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_attr)

        return global_add_pool(x, batch)


class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x_user, x_book, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_book = x_book[edge_label_index[1]]

        z = torch.cat([edge_feat_user, edge_feat_book], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, in_channels_nodes, in_channels_edges, hidden_channels, out_channels, num_layers):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and books:
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.book_emb = torch.nn.Embedding(data["book"].num_nodes, hidden_channels)
        self.genre_emb = torch.nn.Embedding(data["genre"].num_nodes, hidden_channels)
        self.heteroGNN = to_hetero(
            EdgeConvNet(in_channels_nodes, in_channels_edges, hidden_channels, out_channels, num_layers),
            data.metadata(), aggr='sum')
        self.classifier = Classifier(hidden_channels)

    def forward(self, data):
        x_dict = {
            "user": self.user_emb(data["user"].n_id),
            "book": self.book_emb(data["book"].n_id),
            "genre": self.book_emb(data["genre"].n_id),
        }
        x_dict = self.heteroGNN(x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["book"],
            data["user", "review", "book"].edge_label_index,
        )

        return pred, x_dict


if __name__ == "__main__":
    seed_everything(66)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='GraphTransformer',
                        help='Model type for HeteroGNN, options are GraphTransformer, GINE, Spline')
    args = parser.parse_args()

    Dataset = Children(root='.')
    data = Dataset[0]
    print(data)

    num_users = data['user'].num_nodes
    num_books = data['book'].num_nodes
    num_reviews = data['user', 'review', 'book'].num_edges
    num_descriptions = data["book", "description", "genre"].num_edges

    npdata = np.load('children_dataset/emb/review.npy')
    data['user', 'review', 'book'].edge_attr = torch.tensor(npdata).squeeze().float() 
    npdata = np.load('children_dataset/emb/edge_attr_book_genre.npy')
    data['book', 'description', 'genre'].edge_attr = torch.tensor(npdata).squeeze().float()
    del npdata

    # select 4-star or 5-star review as the positive edge
    positive_edges_mask = (data['user', 'review', 'book'].edge_label == 5) | (
            data['user', 'review', 'book'].edge_label == 4)
    data['user', 'review', 'book'].edge_index = data['user', 'review', 'book'].edge_index[:, positive_edges_mask]
    data['user', 'review', 'book'].edge_attr = data['user', 'review', 'book'].edge_attr[positive_edges_mask]

    # Add reverse relations for message passing:
    data = T.ToUndirected()(data)
    del data['book', 'rev_review', 'user'].edge_label  # Remove "reverse" label.
    del data['user', 'review', 'book'].edge_label  # Remove "reverse" label.

    # Perform a link-level split into training, validation, and test edges:
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
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

    model = Model(hidden_channels=256, edge_dim=3072, num_layers=2, model_type=args.model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 10):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data = sampled_data.to(device)
            pred, x_dict = model(sampled_data)
            ground_truth = sampled_data["user", "review", "book"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

        # validation
        if epoch % 1 == 0 and epoch != 0:
            print('Validation begins')
            with torch.no_grad():
                preds = []
                ground_truths = []
                for sampled_data in tqdm.tqdm(test_loader):
                    with torch.no_grad():
                        sampled_data = sampled_data.to(device)
                        pred = model(sampled_data)[0]
                        preds.append(pred)
                        ground_truths.append(sampled_data["user", "review", "book"].edge_label)  # TODO
                        positive_pred = pred[sampled_data["user", "review", "book"].edge_label == 1].cpu().numpy()
                        negative_pred = pred[sampled_data["user", "review", "book"].edge_label == 0].cpu().numpy()
                    pred = torch.cat(preds, dim=0).cpu().numpy()

                ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
                y_label = np.where(pred >= 0.5, 1, 0)
                f1 = f1_score(ground_truth, y_label)
                print(f"F1 score: {f1:.4f}")
                # AUC
                auc = roc_auc_score(ground_truth, pred)
                print(f"Validation AUC: {auc:.4f}")
