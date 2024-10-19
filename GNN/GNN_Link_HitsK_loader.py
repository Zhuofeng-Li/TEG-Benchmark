import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
import argparse
import pickle

import torch
import torch.nn.functional as F

from torch_sparse import SparseTensor

from model.GNN_library import GAT, GINE, GeneralGNN, GraphTransformer
from model.Dataloader import Evaluator, split_edge
from model.GNN_arg import Logger
import wandb
import os


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, train_loader, optimizer, device):
    model.train()
    predictor.train()

    total_loss = total_examples = 0
    for batch in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        h = model(batch.x, batch.adj_t)

        src = batch.edge_label_index.t()[:, 0]
        dst = batch.edge_label_index.t()[:, 1]

        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(
            0, h.size(0), src.size(), dtype=torch.long, device=h.device
        )
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, dataloaders, evaluator, device):
    model.eval()
    predictor.eval()
    (
        train_loader,
        valid_loader,
        valid_negative_loader,
        test_loader,
        test_negative_loader,
    ) = (
        dataloaders["train"],
        dataloaders["valid"],
        dataloaders["valid_negative"],
        dataloaders["test"],
        dataloaders["test_negative"],
    )

    def test_split(dataloader, neg_dataloader, device):
        pos_preds = []
        for batch in dataloader:
            batch = batch.to(device)
            h = model(batch.x, batch.adj_t)
            src = batch.edge_label_index.t()[:, 0]
            dst = batch.edge_label_index.t()[:, 1]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        for batch in neg_dataloader:
            batch = batch.to(device)
            h = model(batch.x, batch.adj_t)
            src = batch.edge_label_index.t()[:, 0]
            dst = batch.edge_label_index.t()[:, 1]
            neg_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0)
        return pos_pred, neg_pred

    pos_train_pred, neg_valid_pred = test_split(
        train_loader, valid_negative_loader, device
    )
    pos_valid_pred, neg_valid_pred = test_split(
        valid_loader, valid_negative_loader, device
    )
    pos_test_pred, neg_test_pred = test_split(test_loader, test_negative_loader, device)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval(
            {
                "y_pred_pos": pos_train_pred,
                "y_pred_neg": neg_valid_pred,
            }
        )[f"hits@{K}"]
        valid_hits = evaluator.eval(
            {
                "y_pred_pos": pos_valid_pred,
                "y_pred_neg": neg_valid_pred,
            }
        )[f"hits@{K}"]
        test_hits = evaluator.eval(
            {
                "y_pred_pos": pos_test_pred,
                "y_pred_neg": neg_test_pred,
            }
        )[f"hits@{K}"]

        results[f"Hits@{K}"] = (train_hits, valid_hits, test_hits)

    return results

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


def gen_loader(args, edge_split, x, edge_index, adj_t):
    train_data = Data(x=x, adj_t=adj_t)
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        edge_label_index=(edge_index),
        edge_label=torch.ones(edge_index.shape[1]),
        batch_size=args.batch_size,
        neg_sampling_ratio=0.0,
        shuffle=True,
    )

    val_edge_label_index = edge_split["valid"]["edge"].t()
    val_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        edge_label_index=(val_edge_label_index),
        edge_label=torch.ones(val_edge_label_index.shape[1]),
        batch_size=args.batch_size,
        neg_sampling_ratio=0.0,
        shuffle=False,
    )

    val_negative_edge_label_index = edge_split["valid"]["edge_neg"].t()
    val_negative_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        edge_label_index=(val_negative_edge_label_index),
        edge_label=torch.zeros(val_negative_edge_label_index.shape[1]),
        batch_size=args.batch_size,
        neg_sampling_ratio=0.0,
        shuffle=False,
    )

    test_edge_label_index = edge_split["test"]["edge"].t()
    test_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        edge_label_index=(test_edge_label_index),
        edge_label=torch.ones(test_edge_label_index.shape[1]),
        batch_size=args.batch_size,
        neg_sampling_ratio=0.0,
        shuffle=False,
    )

    test_negative_edge_label_index = edge_split["test"]["edge_neg"].t()
    test_negative_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        edge_label_index=(test_negative_edge_label_index),
        edge_label=torch.zeros(test_negative_edge_label_index.shape[1]),
        batch_size=args.batch_size,
        neg_sampling_ratio=0.0,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        val_negative_loader,
        test_loader,
        test_negative_loader,
    )

def main():
    parser = argparse.ArgumentParser(description="Link-Prediction PLM/TCL")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--use_node_embedding", action="store_true")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64 * 1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--gnn_model", type=str, help="GNN Model", default="GraphTransformer"
    )
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.08)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--neg_len", type=str, default="5000")
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
        "--path",
        type=str,
        default="data/CSTAG/Photo/LinkPrediction/",
        help="Path to save splitting",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="data/CSTAG/Photo/children.pkl",
        help="Path to load the graph",
    )
    args = parser.parse_args()
    wandb.config = args
    wandb.init(config=args, reinit=True)
    print(args)

    if not os.path.exists(f"{args.path}{args.neg_len}/"):
        os.makedirs(f"{args.path}{args.neg_len}/")

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with open(f"{args.graph_path}", "rb") as file:
        graph = pickle.load(file)

    graph.num_nodes = len(graph.text_nodes)
    edge_split = split_edge(
        graph,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        path=args.path,
        neg_len=args.neg_len,
    )

    x = torch.load(args.use_PLM_node).float()
    edge_feature = torch.load(args.use_PLM_edge)[
        edge_split["train"]["train_edge_feature_index"]
    ].float()

    edge_index = edge_split["train"]["edge"].t()
    adj_t = SparseTensor.from_edge_index(
        edge_index, edge_feature, sparse_sizes=(graph.num_nodes, graph.num_nodes)
    ).t()
    adj_t = adj_t.to_symmetric()

    train_loader, val_loader, val_negative_loader, test_loader, test_negative_loader = (
        gen_loader(args, edge_split, x, edge_index, adj_t)
    )
    dataloaders = {
        "train": train_loader,
        "valid": val_loader,
        "valid_negative": val_negative_loader,
        "test": test_loader,
        "test_negative": test_negative_loader,
    }

    model = gen_model(args, x, edge_feature).to(device)

    predictor = LinkPredictor(
        args.hidden_channels, args.hidden_channels, 1, args.num_layers, args.dropout
    ).to(device)

    evaluator = Evaluator(name="History")
    loggers = {
        "Hits@10": Logger(args.runs, args),
        "Hits@50": Logger(args.runs, args),
        "Hits@100": Logger(args.runs, args),
    }

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()), lr=args.lr
        )

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, train_loader, optimizer, device)

            if epoch % args.eval_steps == 0:
                results = test(
                    model, predictor, dataloaders, evaluator, device
                )
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(
                            f"Run: {run + 1:02d}, "
                            f"Epoch: {epoch:02d}, "
                            f"Loss: {loss:.4f}, "
                            f"Train: {100 * train_hits:.2f}%, "
                            f"Valid: {100 * valid_hits:.2f}%, "
                            f"Test: {100 * test_hits:.2f}%"
                        )
                    print("---")

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(key=key)



if __name__ == "__main__":
    main()
