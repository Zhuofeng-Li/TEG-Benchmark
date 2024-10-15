# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import torch
import pickle
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader
from model.GNN_library import GAT, GINE, GeneralGNN, GraphTransformer
from torch.nn import Linear
from sklearn.metrics import f1_score
import argparse
from torch_geometric.data import Data


class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feat_src = x[edge_label_index[0]]
        edge_feat_dst = x[edge_label_index[1]]

        z = torch.cat([edge_feat_src, edge_feat_dst], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


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
        default="data/CSTAG/Photo/children.pkl",
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

    num_nodes = len(data.text_nodes)
    num_edges = len(data.text_edges)

    x = torch.load(args.use_PLM_node).squeeze().float()
    edge_feature = torch.load(args.use_PLM_edge).squeeze().float()
    data = Data(x=x, edge_index=data.edge_index, edge_attr=edge_feature)

    print(data)

    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=args.val_ratio,  
        num_test=args.test_ratio,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
    )(data)

    # Perform a link-level split into training, validation, and test edges:
    edge_label_index = train_data.edge_label_index
    edge_label = train_data.edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        edge_label_index=(edge_label_index),
        edge_label=edge_label,
        batch_size=args.batch_size,
        shuffle=True,
    )

    edge_label_index = val_data.edge_label_index
    edge_label = val_data.edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=(edge_label_index),
        edge_label=edge_label,
        batch_size=args.batch_size,
        shuffle=False,
    )

    edge_label_index = test_data.edge_label_index
    edge_label = test_data.edge_label
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[20, 10],
        edge_label_index=(edge_label_index),
        edge_label=edge_label,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = gen_model(args, x, edge_feature).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)
    predictor = Classifier(args.hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
            pred = predictor(x, sampled_data.edge_label_index)
            ground_truth = sampled_data.edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
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
                for sampled_data in tqdm.tqdm(test_loader):  # TODO: use val_loader
                    sampled_data = sampled_data.to(device)
                    adj_t = SparseTensor.from_edge_index(
                        sampled_data.edge_index, sampled_data.edge_attr
                    ).t()
                    adj_t = adj_t.to_symmetric()
                    x = model(sampled_data.x, adj_t)
                    pred = predictor(x, sampled_data.edge_label_index)
                    preds.append(pred)
                    ground_truths.append(sampled_data.edge_label)

                preds = torch.cat(preds, dim=0).cpu().numpy()
                ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

                y_label = np.where(preds >= args.threshold, 1, 0)
                f1 = f1_score(ground_truth, y_label)
                print(f"F1 score: {f1:.4f}")

                # AUC
                auc = roc_auc_score(ground_truth, preds)
                print(f"Validation AUC: {auc:.4f}")
