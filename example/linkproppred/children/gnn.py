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
from torch.nn import Linear
import pickle
from info_nce import InfoNCE, info_nce
from sklearn.metrics import f1_score


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
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)  # TODO
            x_dict = {key: x.relu() for key, x in x_dict.items()} if i != len(
                self.convs) - 1 else x_dict  # TODO: relu depends on downstream tasks
        return x_dict


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
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
    def __init__(self, hidden_channels, edge_dim, num_layers):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and books:
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.book_emb = torch.nn.Embedding(data["book"].num_nodes, hidden_channels)
        self.genre_emb = torch.nn.Embedding(data["genre"].num_nodes, hidden_channels)
        self.heteroGNN = HeteroGNN(hidden_channels, edge_dim, num_layers)
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


def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return (x / norm).tolist()
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return (np.where(norm == 0, x, x / norm)).tolist()


def get_LLM_path_embeddings(data, edge_embedding):
    user_n_id = data['user'].n_id
    book_n_id = data['book'].n_id
    edge_label_index = data['user', 'review', 'book'].edge_label_index
    original_edge_label_index = torch.stack([user_n_id[edge_label_index[0]], book_n_id[edge_label_index[1]]])

    user = original_edge_label_index[0].tolist()
    book = original_edge_label_index[1].tolist()
    user_book = [str(a) + '|' + str(b) for a, b in zip(user, book)]

    z = torch.tensor([edge_embedding[key][:128] for key in user_book]).cuda()  # TODO: embedding dim
    return z


def get_GNN_path_embeddings(x_dict, edge_label_index):
    x_user = x_dict['user']
    x_book = x_dict['book']
    edge_feat_user = x_user[edge_label_index[0]]
    edge_feat_book = x_book[edge_label_index[1]]

    z = torch.cat([edge_feat_user, edge_feat_book], dim=-1)
    return z


def eval_hits(y_pred_pos, y_pred_neg, K):
    kth_score_in_negative_edges = np.sort(y_pred_neg)[-K]
    hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / y_pred_pos.shape[0]
    return hitsK


if __name__ == "__main__":

    seed_everything(66)
    Dataset = Goodreads_children_genre(root='.')
    data = Dataset[0]
    # print(data)

    num_users = data['user'].num_nodes
    num_books = data['book'].num_nodes
    num_reviews = data['user', 'review', 'book'].num_edges
    num_descriptions = data["book", "description", "genre"].num_edges

    with open('children_genre/raw/0.15_train_embeddings.pkl', 'rb') as f:
        edge_embedding = pickle.load(f)

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
        num_val=0.85,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        # add_negative_train_samples=False,
        edge_types=[('user', 'review', 'book')],
        rev_edge_types=[('book', 'rev_review', 'user')],
    )(data)

    edge_label_index = train_data["user", "review", "book"].edge_label_index
    edge_label = train_data["user", "review", "book"].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        # neg_sampling_ratio=2.0,
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
        shuffle=True,  # To calculate Hits@N
    )

    model = Model(hidden_channels=64, edge_dim=3072, num_layers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    info_nce = InfoNCE()
    eta = 0
    print(f'eta {eta}')
    for epoch in range(1, 30):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data = sampled_data.to(device)
            pred, x_dict = model(sampled_data)
            # contrastive learning
            LLM_path_embeddings = get_LLM_path_embeddings(sampled_data, edge_embedding)
            GNN_path_embeddings = get_GNN_path_embeddings(x_dict,
                                                          sampled_data["user", "review", "book"].edge_label_index)
            info_nce_loss = info_nce(GNN_path_embeddings, LLM_path_embeddings)
            ground_truth = sampled_data["user", "review", "book"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth) + eta * info_nce_loss
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
                        preds.append(model(sampled_data)[0])
                        ground_truths.append(sampled_data["user", "review", "book"].edge_label)  # TODO
                pred = torch.cat(preds, dim=0).cpu().numpy()
                ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
                # AUC
                auc = roc_auc_score(ground_truth, pred)
                print(f"Validation AUC: {auc:.4f}")
                # F1
                y_label = np.where(pred >= 0.5, 1, 0)
                f1 = f1_score(ground_truth, y_label)
                print(f"F1 score: {f1:.4f}")
