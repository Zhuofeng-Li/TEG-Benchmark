import torch
import pickle
from torch_sparse import SparseTensor
import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric import seed_everything
from torch_geometric.loader import NeighborLoader
from model.GNN_library import GAT, GINE, GeneralGNN, GraphTransformer
from torch.nn import Linear
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import argparse
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer


class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x


def gen_model(args, x, edge_feature):
    if args.gnn_model == "GAT":
        model = GAT(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.heads,
            args.dropout,
        )
    elif args.gnn_model == "GraphTransformer":
        model = GraphTransformer(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
        )
    elif args.gnn_model == "GINE":
        model = GINE(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
        )
    elif args.gnn_model == "GeneralGNN":
        model = GeneralGNN(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
        )
    else:
        raise ValueError("Not implemented")
    return model


def train(model, predictor, train_loader, optimizer, criterion):
    model.train()
    predictor.train()

    total_loss = total_examples = 0
    for batch in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        h = model(batch.x, batch.adj_t)[: batch.batch_size]
        pred = predictor(h)
        loss = criterion(pred, batch.y[: batch.batch_size].float())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    return total_examples, total_loss


def process_data(args, device, data):  # TODO: process all data here
    num_nodes = len(data.text_nodes)
    data.num_nodes = num_nodes

    product_indices = torch.tensor(
        [
            i for i, label in enumerate(data.text_node_labels) if label != -1
        ]  # TODO: check the label in each dataset
    ).long()
    product_labels = [label for label in data.text_node_labels if label != -1]
    mlb = MultiLabelBinarizer()
    product_binary_labels = mlb.fit_transform(product_labels)
    y = torch.zeros(num_nodes, product_binary_labels.shape[1]).float()
    y[product_indices] = torch.tensor(product_binary_labels).float()
    y = y.to(device)  # TODO: check the label

    train_ratio = 1 - args.test_ratio - args.val_ratio
    val_ratio = args.val_ratio

    num_products = product_indices.shape[0]
    train_idx = product_indices[: int(num_products * train_ratio)]
    val_idx = product_indices[
        int(num_products * train_ratio) : int(num_products * (train_ratio + val_ratio))
    ]
    test_idx = product_indices[
        int(product_indices.shape[0] * (train_ratio + val_ratio)) :
    ]

    x = torch.load(args.use_PLM_node).squeeze().float()
    edge_feature = torch.load(args.use_PLM_edge).squeeze().float()

    edge_index = data.edge_index
    adj_t = SparseTensor.from_edge_index(
        edge_index, edge_feature, sparse_sizes=(data.num_nodes, data.num_nodes)
    ).t()
    adj_t = adj_t.to_symmetric()
    node_split = {"train": train_idx, "val": val_idx, "test": test_idx}
    return node_split, x, edge_feature, adj_t, y


def test(model, predictor, test_loader, args):
    print("Validation begins")
    with torch.no_grad():
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(test_loader):
            batch = batch.to(device)
            h = model(batch.x, batch.adj_t)[: batch.batch_size]
            pred = predictor(h)
            ground_truth = batch.y[: batch.batch_size]
            preds.append(pred)
            ground_truths.append(ground_truth)

        preds = torch.cat(preds, dim=0).cpu().numpy()
        ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()

        y_label = (preds > args.threshold).astype(int)
        f1 = f1_score(ground_truths, y_label, average="weighted")
        print(f"F1 score: {f1:.4f}")

        data_type = args.graph_path.split("/")[-1]
        if data_type not in [
            "twitter.pkl",
            "reddit.pkl",
            "citation.pkl",
        ]:  # TODO: check metric in
            auc = roc_auc_score(ground_truths, preds, average="micro")
            print(f"Validation AUC: {auc:.4f}")
        else:
            accuracy = accuracy_score(ground_truths, y_label)
            print(f"Validation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    # TODO: use wandb as log
    seed_everything(66)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--gnn_model",
        "-gm",
        type=str,
        default="GAT",
        help="Model type for HeteroGNN, options are GraphTransformer, GINE, GraphSAGE, GeneralConv, MLP, EdgeConv,RevGAT",
    )
    parser.add_argument(
        "--use_PLM_node",
        type=str,
        default="data/CSTAG/Photo/Feature/children_gpt_node.pt",
        help="Use LM embedding as node feature",
    )
    parser.add_argument(
        "--use_PLM_edge",
        type=str,
        default="data/CSTAG/Photo/Feature/children_gpt_edge.pt",
        help="Use LM embedding as edge feature",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="../../TEG-Datasets/goodreads_children/processed/children.pkl",  # "data/CSTAG/Photo/children.pkl",
        help="Path to load the graph",
    )
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with open(f"{args.graph_path}", "rb") as f:
        data = pickle.load(f)

    node_split, x, edge_feature, adj_t, y = process_data(args, device, data)
    data = Data(x=x, adj_t=adj_t, y=y)

    train_loader = NeighborLoader(
        data,
        input_nodes=node_split["train"],
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=node_split["val"],
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=node_split["test"],
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=False,
    )  # TODO: move `num_neighbors` to args

    model = gen_model(args, x, edge_feature)
    model = model.to(device)
    predictor = Classifier(args.hidden_channels, y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 1 + args.epochs):
        total_examples, total_loss = train(
            model, predictor, train_loader, optimizer, criterion
        )
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

        # validation
        if epoch % args.eval_steps == 0:
            test(model, predictor, test_loader, args)
