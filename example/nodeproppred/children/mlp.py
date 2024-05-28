import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from models import SAGEEdgeConv, EdgeConvConv
import numpy as np
import torch
import torch_geometric.transforms as T
import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.conv import GeneralConv
from torch.nn import Linear
from sklearn.metrics import f1_score
import argparse

from TAG.linkproppred.children import Children


class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels / 4)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x_book):
        z = x_book
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return torch.sigmoid(z)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_dim, num_layers, model_type):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and books:
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.book_emb = torch.nn.Embedding(data["book"].num_nodes, hidden_channels)
        self.genre_emb = torch.nn.Embedding(data["genre"].num_nodes, hidden_channels)
        self.classifier = Classifier(hidden_channels, out_channels)

    def forward(self, data):
        x_dict = {
            "user": self.user_emb(data["user"].n_id),
            "book": self.book_emb(data["book"].n_id),
            "genre": self.book_emb(data["genre"].n_id),
        }
        pred = self.classifier(x_dict["book"])

        return pred, x_dict


if __name__ == '__main__':
    Goodreads_dataset = Children(root='.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', '-dt', type=str, default='children',
                        help='goodreads dataset type for children, crime, history, mystery')
    parser.add_argument('--emb_type', '-et', type=str, default='GPT-3.5-TURBO',
                        help='embedding type for HeteroGNN, options are GPT-3.5-TURBO, Bert, Angle, None')
    parser.add_argument('--model_type', '-mt', type=str, default='GraphTransformer',
                        help='Model type for HeteroGNN, options are GraphTransformer, GINE, Spline')
    args = parser.parse_args()

    Dataset = Children(root='.')
    data = Dataset[0]

    num_reviews = data['user', 'review', 'book'].num_edges

    # load emb
    if args.emb_type == 'GPT-3.5-TURBO':
        encoded_text = np.load('children_dataset/emb/review.npy')
        data['user', 'reviews', 'book'].edge_attr = torch.tensor(encoded_text).squeeze().float()
    elif args.emb_type == 'None':
        data['user', 'review', 'book'].edge_attr = torch.randn(num_reviews, 256).squeeze().float()

    data = T.ToUndirected()(data)  # To message passing

    # dataloader
    readers_samples, books_samples, batch_size = 1024, 1024, 1024

    train_loader = HGTLoader(
        data,
        num_samples={'user': [readers_samples], 'book': [books_samples]},
        batch_size=batch_size,
        input_nodes=('book', data['book'].train_mask),
    )

    val_loader = HGTLoader(
        data,
        num_samples={'user': [readers_samples], 'book': [books_samples]},
        batch_size=batch_size,
        input_nodes=('book', data['book'].val_mask),
    )

    test_loader = HGTLoader(
        data,
        num_samples={'user': [readers_samples], 'book': [books_samples]},
        batch_size=batch_size,
        input_nodes=('book', data['book'].test_mask),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = Model(hidden_channels=256, out_channels=data.num_classes, edge_dim=3072, num_layers=2,
                  model_type=args.model_type)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # weight for unbalanced classes
    weight = data['book'].y.long().sum(0)
    weight = weight.max() / weight

    criterion = torch.nn.BCELoss(weight=weight)
    criterion = criterion.to(device)

    loss_list = []
    acc_list = []
    prc_list = []
    mrr_list = []
    ndcg_list = []
    micro_f1_list = []
    macro_f1_list = []

    for epoch in range(10):
        model.train()
        total_examples = total_loss = 0

        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            batch_size = batch['book'].batch_size

            out, x_dict = model(batch)

            loss = criterion(out, batch['book'].y.squeeze())
            loss.backward()
            optimizer.step()

            total_examples += batch_size
            total_loss += float(loss) * batch_size

        # validation
        if epoch % 1 == 0 and epoch != 0:
            print('Validation begins')

            model.eval()
            with torch.no_grad():
                preds = []
                ground_truths = []
                for sampled_data in tqdm.tqdm(test_loader):
                    with torch.no_grad():
                        sampled_data = sampled_data.to(device)
                        pred = model(sampled_data)[0]
                        preds.append(pred)
                        ground_truths.append(sampled_data["book"].y.squeeze())
                    pred = torch.cat(preds, dim=0).cpu().numpy()

                ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
                y_label = np.where(pred >= 0.5, 1, 0)
                f1 = f1_score(ground_truth, y_label, average='weighted')
                print(f"F1 score: {f1:.4f}")
                # AUC
                auc = roc_auc_score(ground_truth, pred)
                print(f"Validation AUC: {auc:.4f}")
