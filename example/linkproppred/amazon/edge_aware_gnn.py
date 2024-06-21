import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from models import SAGEEdgeConv, EdgeConvConv
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.conv import GeneralConv
from torch.nn import Linear, ModuleList
from sklearn.metrics import f1_score
import argparse

from TAG.linkproppred.app import Amazon_Apps
from TAG.linkproppred.movie import Amazon_Movies


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, edge_dim, num_layers, model_type):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        if model_type == 'GraphSAGE':
            self.conv = SAGEEdgeConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        elif model_type == 'GraphTransformer':
            self.conv = TransformerConv((-1, -1), hidden_channels, edge_dim=edge_dim)
        elif model_type == 'GINE':
            self.conv = GINEConv(Linear(hidden_channels, hidden_channels), train_eps=True, edge_dim=edge_dim)
        elif model_type == 'EdgeConv':
            self.conv = EdgeConvConv(Linear(2 * hidden_channels + edge_dim, hidden_channels), train_eps=True, edge_dim=edge_dim)
        elif model_type == 'GeneralConv':
            self.conv = GeneralConv((-1, -1), hidden_channels, in_edge_channels=edge_dim)
        else:
            raise NotImplementedError('Model type not implemented')
        
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: self.conv for edge_type in data.edge_types
            }, aggr='sum')

            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()} if i != len(self.convs) - 1 else x_dict
        return x_dict


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
    def __init__(self, hidden_channels, edge_dim, num_layers, model_type):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and books:
        self.model_type = model_type
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.book_emb = torch.nn.Embedding(data["book"].num_nodes, hidden_channels)
        if model_type != 'MLP':
            self.heteroGNN = HeteroGNN(hidden_channels, edge_dim, num_layers, model_type=model_type)
        self.classifier = Classifier(hidden_channels)

    def forward(self, data):
        x_dict = {
            "user": self.user_emb(data["user"].n_id),
            "book": self.book_emb(data["book"].n_id),
        }
        
        if self.model_type != 'MLP':
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
    parser.add_argument('--data_type', '-dt', type=str, default='app',
                        help='data type for datasets, options are app,movie')
    parser.add_argument('--emb_type', '-et', type=str, default='None',
                        help='Model type for HeteroGNN, options are GPT-3.5-TURBO, Bert,Large_Bert None')
    parser.add_argument('--model_type', '-mt', type=str, default='GraphSAGE',
                        help='Model type for HeteroGNN, options are GraphTransformer, GINE, GraphSAGE, GeneralConv, MLP, EdgeConv')
    args = parser.parse_args()

    if args.data_type == 'app':
        Dataset = Amazon_Apps(root=f'{args.data_type}')
    elif args.data_type == 'movie':
        Dataset = Amazon_Movies(root=f'{args.data_type}')
    else:
        raise NotImplementedError('Dataset not implemented')
    data = Dataset[0]
    print(data)

    num_users = data['user'].num_nodes
    num_books = data['book'].num_nodes
    num_reviews = data['user', 'review', 'book'].num_edges
    #num_descriptions = data["book", "description", "genre"].num_edges
    
    # load emb 
    if args.emb_type != 'None':
        npdata = np.load(f'{args.data_type}/{args.data_type}_dataset/emb/{args.data_type}_reviews_{args.emb_type}.npy')
        data['user', 'review', 'book'].edge_attr = torch.tensor(npdata).squeeze().float()
        del npdata
        if args.emb_type == 'GPT-3.5-TURBO':
            edge_dim = 128
        elif args.emb_type == 'Bert':
            edge_dim = 512
        elif args.emb_type == 'Large_Bert':
            edge_dim = 512
        else:
            raise NotImplementedError('Embedding type not implemented')
    else:
        data['user', 'review', 'book'].edge_attr = torch.randn(num_reviews, 512).squeeze().float()
        edge_dim = 512

    # select 4-star or 5-star review as the positive edge
    positive_edges_mask = data['user', 'review', 'book'].edge_label == 5
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

    model = Model(hidden_channels=256, edge_dim=edge_dim, num_layers=2, model_type=args.model_type)
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
                        ground_truths.append(sampled_data["user", "review", "book"].edge_label)  
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
