import numpy as np
import torch
import pickle
import torch.nn.functional as F
import torch_geometric.transforms as T
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
        self.lin1 = Linear(hidden_channels, hidden_channels // 4)
        self.lin2 = Linear(hidden_channels // 4, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x


if __name__ == "__main__":
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
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with open(f"{args.graph_path}", "rb") as f:
        data = pickle.load(f)

    num_nodes = len(data.text_nodes)
    num_edges = len(data.text_edges)
    data.num_nodes = num_nodes

    product_indices = torch.tensor(
        [i for i, label in enumerate(data.text_node_labels) if label != -1]
    )
    user_indices = torch.tensor(
        [i for i, label in enumerate(data.text_node_labels) if label == -1]
    )
    product_labels = [label for label in data.text_node_labels if label != -1]
    real_id = 0
    n_id_to_index = {}
    for i, label in enumerate(data.text_node_labels):
        if label != -1:
            n_id_to_index[i] = real_id
            real_id += 1

    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(product_labels)
    y = torch.tensor(binary_labels).long()

    train_ratio = 1 - args.test_ratio - args.val_ratio
    val_ratio = args.val_ratio

    num_train_products = int(len(product_labels) * train_ratio)
    num_val_products = int(len(product_labels) * val_ratio)
    num_test_products = int(len(product_labels)) - num_train_products - num_val_products

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[product_indices[:num_train_products]] = 1  # TODO: wrong 
    train_mask[user_indices] = 1
    val_mask[
        product_indices[num_train_products + num_val_products]
    ] = 1  # TODO:how to set val_mask
    val_mask[user_indices] = 1
    test_mask[product_indices[-num_test_products:]] = 1  # TODO: random shuffle

    num_classes = len(mlb.classes_)

    x = torch.load(args.use_PLM_node).squeeze().float()
    edge_feature = torch.load(args.use_PLM_edge).squeeze().float()
    data = Data(
        x=x,
        y=y,
        edge_index=data.edge_index,
        edge_attr=edge_feature,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=num_classes,
    )

    print(data)

    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=[10, 10],
        batch_size=1024,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=data.val_mask,
        num_neighbors=[10, 10],
        batch_size=1024,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=data.test_mask,
        num_neighbors=[10, 10],
        batch_size=1024,
        shuffle=False,
    )

    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=[10, 10],
        batch_size=1024,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=data.val_mask,
        num_neighbors=[10, 10],
        batch_size=1024,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=data.test_mask,
        num_neighbors=[10, 10],
        batch_size=1024,
        shuffle=False,
    )

    if args.gnn_model == "GAT":
        model = GAT(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.heads,
            args.dropout,
        ).to(device)
    elif args.gnn_model == "GraphTransformer":
        model = GraphTransformer(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
        ).to(device)
    elif args.gnn_model == "GINE":
        model = GINE(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
        ).to(device)
    elif args.gnn_model == "GeneralGNN":
        model = GeneralGNN(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
        ).to(device)
    else:
        raise ValueError("Not implemented")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)
    predictor = Classifier(args.hidden_channels, data.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    data_type = args.graph_path.split("/")[-1]

    for epoch in range(1, 1 + args.epochs):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data = sampled_data.to(device)
            adj_t = SparseTensor.from_edge_index(
                sampled_data.edge_index, sampled_data.edge_attr
            ).t()
            adj_t = adj_t.to_symmetric()
            x = model(sampled_data.x, adj_t)
            pred = predictor(x)

            new_indices = [
                n_id_to_index.get(n_id.item(), -1) for n_id in sampled_data.n_id
            ]
            valid_indices = [index for index in new_indices if index != -1]
            judge = [True if x != -1 else False for x in new_indices]
            if valid_indices:
                true = sampled_data.y[torch.tensor(valid_indices)].to(device)
                out_filtered = pred[torch.tensor(judge, dtype=torch.bool)]
            else:
                raise ValueError("No valid indices found for sampled_data.y.")

            loss = criterion(out_filtered, true.float())
            # loss = criterion(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

        # validation
        if epoch % args.eval_steps == 0 and epoch != 0:
            print("Validation begins")
            with torch.no_grad():
                preds = []
                ground_truths = []
                for sampled_data in tqdm.tqdm(test_loader):
                    sampled_data = sampled_data.to(device)
                    adj_t = SparseTensor.from_edge_index(
                        sampled_data.edge_index, sampled_data.edge_attr
                    ).t()
                    adj_t = adj_t.to_symmetric()
                    x = model(sampled_data.x, adj_t)
                    pred = predictor(x)
                    new_indices = [
                        n_id_to_index.get(n_id.item(), -1) for n_id in sampled_data.n_id
                    ]
                    valid_indices = [index for index in new_indices if index != -1]
                    judge = [True if x != -1 else False for x in new_indices]
                    if valid_indices:
                        true = sampled_data.y[torch.tensor(valid_indices)].to(device)
                        out_filtered = pred[torch.tensor(judge, dtype=torch.bool)]

                    preds.append(out_filtered)
                    ground_truths.append(true)

                preds = torch.cat(preds, dim=0).cpu().numpy()
                ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

                y_label = (preds > args.threshold).astype(int)
                f1 = f1_score(ground_truth, y_label, average="weighted")
                print(f"F1 score: {f1:.4f}")
                if data_type not in ["twitter.pkl", "reddit.pkl", "citation.pkl"]:
                    auc = roc_auc_score(ground_truth, preds, average="micro")
                    print(f"Validation AUC: {auc:.4f}")
                else:
                    accuracy = accuracy_score(ground_truth, y_label)
                    print(f"Validation Accuracy: {accuracy:.4f}")
