import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import Linear, HeteroConv, TransformerConv
import torch_geometric.transforms as T

from children.goodreads_children import Goodreads_children


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_dim, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: TransformerConv((-1, -1), hidden_channels, edge_dim=edge_dim) for edge_type in
                data.edge_types
            }, aggr='sum')

            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        res = self.lin(x_dict['book'])
        return torch.sigmoid(res)


@torch.no_grad()
def init_params():
    # Initialize lazy parameters via forwarding a single batch to the model:
    batch = next(iter(train_loader))
    batch = batch.to(device, 'edge_attr', 'edge_index')
    model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)


def train():
    model.train()
    total_examples = total_loss = 0
    criterion = torch.nn.BCELoss(weight=weight)
    criterion = criterion.to(device)

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device, 'edge_index', 'edge_attr', 'x', 'y')
        batch_size = batch['book'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)

        loss = criterion(out, batch['book'].y[:batch_size].squeeze())
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()
    labels = np.zeros(10)[None, :]
    count = 0
    metrics_total = defaultdict(float)
    preds = np.zeros(10)[None, :]
    for batch in loader:
        batch = batch.to(device, 'edge_index', 'edge_attr', 'x', 'y')
        batch_size = batch['book'].batch_size

        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        predictions = torch.argmax(out, dim=-1).cpu().detach().numpy()
        out = out.cpu().detach().numpy()
        metrics, pred = evaluator(out, batch['book'].y[:batch_size].squeeze().cpu().detach().numpy(), predictions)
        labels = np.concatenate((labels, batch['book'].y[:batch_size].squeeze().cpu().detach().numpy()))

        for k, v in metrics.items():
            metrics_total[k] += v
        count += 1
        preds = np.concatenate((preds, pred), axis=0)

    for key in metrics_total:
        metrics_total[key] /= count
        print("{}:{}".format(key, metrics_total[key]))

    labels = labels[1:, :]
    preds = preds[1:, :]
    metrics_total['micro_f1'] = f1_score(labels, preds, average='micro')
    metrics_total['macro_f1'] = f1_score(labels, preds, average='macro')
    print("{}:{}".format('micro_f1', metrics_total['micro_f1']))
    print("{}:{}".format('macro_f1', metrics_total['macro_f1']))
    return metrics_total['acc'], metrics_total['prc'], metrics_total['mrr'], metrics_total['ndcg'], metrics_total[
        'micro_f1'], metrics_total['macro_f1']


def acc(y_true, y_hat):  # Not used
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k) + 1e-10
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true + 1e-10)


def evaluator(scores, labels, predictions):
    prc = sum([labels[i][predictions[i]].item() for i in range(labels.shape[0])]) / labels.shape[0]
    mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
    mrr = np.mean(mrr_all)
    ndcg_all = [ndcg_score(labels[i], scores[i], 3) for i in range(labels.shape[0])]
    ndcg = np.mean(ndcg_all)
    preds = (scores > 0.5).astype(int)
    acc = (preds == labels).sum().sum() / labels.shape[0] / labels.shape[1]
    return {
        "main": prc,
        "prc": prc,
        "mrr": mrr,
        "ndcg": ndcg,
        "acc": acc,
    }, (scores > 0.5).astype(int)


review_embedding_path = '/root/autodl-tmp/Graph_LLM_link_predcition-main/NELL/get_embedding_llma/review_embedding_13b.npy'
epcho = 1000

if __name__ == '__main__':

    print(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

    Goodreads_dataset = Goodreads_children(root='..')
    data = Goodreads_dataset[0]
    encoded_text = np.load(review_embedding_path)
    data['user', 'review', 'book'].edge_attr = torch.tensor(encoded_text).squeeze().float()

    data = T.ToUndirected()(data)  # To message passing

    readers_samples = 1024
    books_samples = 1024
    batch_size = 1024

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
    data = data.to(device, 'x', 'y')  # TODO: mask need device?

    # weight for unbalanced classes
    weight = data['book'].y.long().sum(0)
    weight = weight.max() / weight

    model = HeteroGNN(hidden_channels=64, out_channels=data.num_classes, edge_dim=5120, num_layers=2).to(device)
    init_params()  # Initialize parameters.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss_list = []
    acc_list = []
    prc_list = []
    mrr_list = []
    ndcg_list = []
    micro_f1_list = []
    macro_f1_list = []

    for epoch in range(1, epcho):
        loss = train()

        x1, x2, x3, x4, x5, x6 = test(val_loader)
        loss_list.append(loss)
        acc_list.append(x1)
        prc_list.append(x2)
        mrr_list.append(x3)
        ndcg_list.append(x4)
        micro_f1_list.append(x5)
        macro_f1_list.append(x6)

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val acc: {x1:.4f}')

    sub_plot = [acc_list, prc_list, mrr_list, ndcg_list, micro_f1_list, macro_f1_list]
    title = ['acc', 'prc', 'mrr', 'ndcg', 'micro_f1', 'macro_f1']

    plt.figure(figsize=(10, 10))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(sub_plot[i])
        plt.title(title[i])
    plt.show()
