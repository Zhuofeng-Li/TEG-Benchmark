import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from children_genre.goodreads_children_genre import Goodreads_children_genre
from sklearn.metrics import roc_auc_score
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv
# from transformer_conv import TransformerConv
from torch_geometric.nn import TransformerConv


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, edge_dim, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: TransformerConv((-1, -1), hidden_channels, edge_dim=edge_dim) for edge_type in
                data.edge_types
            }, aggr='sum')

            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()} if i != len(
                self.convs) - 1 else x_dict  # TODO: relu depends on downstream tasks
        return x_dict


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user, x_book, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_book = x_book[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_book).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, edge_dim, num_layers):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and books:
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.book_emb = torch.nn.Embedding(data["book"].num_nodes, hidden_channels)
        self.genre_emb = torch.nn.Embedding(data["genre"].num_nodes, hidden_channels)
        self.heteroGNN = HeteroGNN(hidden_channels, edge_dim, num_layers)
        self.classifier = Classifier()

    def forward(self, data):
        x_dict = {
            "user": self.user_emb(data["user"].n_id),
            "book": self.book_emb(data["book"].n_id),
            "genre": self.book_emb(data["genre"].n_id),
        }
        x_dict = self.heteroGNN(x_dict, data.edge_index_dict, data.edge_attr_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["book"],
            data["user", "review", "book"].edge_label_index,
        )

        return pred


def acc(y_true, y_hat):
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
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


if __name__ == "__main__":

    seed_everything(66)
    Dataset = Goodreads_children_genre(root='.')
    data = Dataset[0]
    # print(data)

    num_users = data['user'].num_nodes
    num_books = data['book'].num_nodes
    num_reviews = data['user', 'review', 'book'].num_edges
    num_descriptions = data["book", "description", "genre"].num_edges

    npdata = np.load('children_genre/raw/review.npy')
    data['user', 'review', 'book'].edge_attr = torch.tensor(npdata).squeeze().float()
    npdata = np.load('children_genre/raw/edge_attr_book_genre.npy')
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
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=[('user', 'review', 'book')],
        rev_edge_types=[('book', 'rev_review', 'user')],
    )(data)

    edge_label_index = train_data["user", "review", "book"].edge_label_index
    edge_label = train_data["user", "review", "book"].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        edge_label_index=(("user", "review", "book"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
        shuffle=True,
    )

    edge_label_index = val_data["user", "review", "book"].edge_label_index
    edge_label = val_data["user", "review", "book"].edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=(("user", "review", "book"), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * 128,
        shuffle=False,
    )

    edge_label_index = test_data["user", "review", "book"].edge_label_index
    edge_label = test_data["user", "review", "book"].edge_label
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[20, 10],
        edge_label_index=(("user", "review", "book"), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * 128,
        shuffle=False,
    )

    model = Model(hidden_channels=64, edge_dim=3072, num_layers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 50):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data["user", "review", "book"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

    # validation
    preds = []
    ground_truths = []
    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["user", "review", "book"].edge_label)  # TODO
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print(f"Validation AUC: {auc:.4f}")

    # test
    preds = []
    ground_truth = []
    count = accc = auc = mrr = ndcg = 0
    for sampled_data in tqdm.tqdm(test_loader):
        with torch.no_grad():
            sampled_data.to(device)
            modelpreds = model(sampled_data)
            ground_truths = sampled_data["user", "review", "book"].edge_label
            ground_truth.append(ground_truths.cpu().numpy())
            preds.append(modelpreds.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    auc = roc_auc_score(ground_truth, preds)
    mrr = mrr_score(ground_truth, preds)
    ndcg = ndcg_score(ground_truth, preds)
    predictions = (preds > 0.5).astype(int)
    accc = (np.sum((predictions == ground_truth)) / ground_truth.shape[0]).item()
    print(f"Test AUC: {auc:.4f}")
    print(f"Test MRR: {mrr:.4f}")
    print(f"Test NDCG: {ndcg:.4f}")
    print(f"Test ACC: {accc:.4f}")
