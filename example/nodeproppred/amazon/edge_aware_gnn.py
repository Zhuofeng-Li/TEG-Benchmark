import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from models import SAGEEdgeConv, EdgeConvConv
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from torch.nn import Linear
from sklearn.metrics import roc_auc_score
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.conv import GeneralConv
from torch.nn import Linear, ModuleList
from torch_geometric.loader import HGTLoader
from sklearn.metrics import f1_score, accuracy_score
import argparse
import warnings
warnings.filterwarnings("ignore")

from TAG.nodeproppred.app import Amazon_Apps
from TAG.nodeproppred.movie import Amazon_Movies


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
            conv = HeteroConv({edge_type: self.conv for edge_type in data.edge_types}, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()} if i != len(self.convs) - 1 else x_dict
        return x_dict


class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 4)
        self.lin2 = Linear(hidden_channels // 4, out_channels)

    def forward(self, x_book):
        z = x_book
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return torch.sigmoid(z)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_dim, num_layers, model_type):
        super().__init__()
        self.model_type = model_type
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.book_emb = torch.nn.Embedding(data["book"].num_nodes, hidden_channels)
        if model_type != 'MLP':
            self.heteroGNN = HeteroGNN(hidden_channels, edge_dim, num_layers, model_type=model_type)
        self.classifier = Classifier(hidden_channels, out_channels)

    def forward(self, data):
        x_dict = {
            "user": self.user_emb(data["user"].n_id),
            "book": self.book_emb(data["book"].n_id),
        }
        if self.model_type != 'MLP':
            x_dict = self.heteroGNN(x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict)
        pred = self.classifier(x_dict["book"])
        return pred, x_dict


if __name__ == '__main__':
    seed_everything(66)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', '-dt', type=str, default='app', help='Dataset type for app,movie')
    parser.add_argument('--emb_type', '-et', type=str, default='None', help='Embedding type, options are GPT-3.5-TURBO, Bert, Large_Bert, None')
    parser.add_argument('--model_type', '-mt', type=str, default='GraphSAGE', help='Model type, options are MLP, GraphSAGE, GraphTransformer, GINE, GeneralConv, EdgeConv')
    args = parser.parse_args()

    if args.data_type == "app":
        Dataset = Amazon_Apps(root=f'{args.data_type}')
    elif args.data_type == "movie":
        Dataset = Amazon_Movies(root=f'{args.data_type}')
    else:
        raise NotImplementedError('Dataset not implemented')
    
    data = Dataset[0]
    print(data)

    num_reviews = data['user', 'review', 'book'].num_edges

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

    data = T.ToUndirected()(data)

    readers_samples, books_samples, batch_size = 1024, 1024, 1024

    train_loader = HGTLoader(
        data,
        num_samples={'user': [readers_samples], 'book': [books_samples]},
        input_nodes=('book', data['book'].train_mask),
        batch_size=1024,
        shuffle=True
    )

    val_loader = HGTLoader(
        data,
        num_samples={'user': [readers_samples], 'book': [books_samples]},
        input_nodes=('book', data['book'].val_mask),
        batch_size=1024,
        shuffle=False
    )

    test_loader = HGTLoader(
        data,
        num_samples={'user': [readers_samples], 'book': [books_samples]},
        input_nodes=('book', data['book'].test_mask),
        batch_size=1024,
        shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = Model(hidden_channels=256, out_channels=data.num_classes, edge_dim=edge_dim, num_layers=2, model_type=args.model_type)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    weight = data['book'].y.long().sum(0)
    #weight = weight.max() / weight
    criterion = torch.nn.BCELoss(weight=weight).to(device)

    for epoch in range(1, 10):
        model.train()
        total_examples = total_loss = 0

        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            out, x_dict = model(batch)

            batch_indices = batch['book'].n_id.cpu()
            batch_labels = data['book'].y[batch_indices].to(device)
            #print(out)
            loss = criterion(out, batch_labels.squeeze())
            loss.backward()
            optimizer.step()

            batch_size = out.shape[0]
            total_examples += batch_size

            total_loss += float(loss) * batch_size

        # Validation
        if epoch % 1 == 0 and epoch != 0:
            print('Validation begins')
            model.eval()
            with torch.no_grad():
                preds = []
                ground_truths = []
                for sampled_data in tqdm.tqdm(test_loader):
                    sampled_data = sampled_data.to(device)
                    pred = model(sampled_data)[0]
                    preds.append(pred)
                    batch_indices = sampled_data['book'].n_id.cpu()
                    ground_truths.append(data['book'].y[batch_indices].to(device).squeeze())

                pred = torch.cat(preds, dim=0).cpu().numpy()
                ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

                y_label = np.where(pred >= 0.5, 1, 0)

                f1 = f1_score(ground_truth, y_label, average='weighted')
                print(f"F1 score: {f1:.4f}")

                micro_auc = roc_auc_score(ground_truth, pred, average='micro')
                print(f"Validation micro AUC: {micro_auc:.4f}")

                ground_truth_flat = ground_truth.ravel()
                y_label_flat = y_label.ravel()
                micro_accuracy = accuracy_score(ground_truth_flat, y_label_flat)
                print(f"Validation micro ACC : {micro_accuracy:.4f}")
