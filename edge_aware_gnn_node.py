import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from models.sage_edge_conv import SAGEEdgeConv
from models.edge_conv import EdgeConvConv
import torch.nn.functional as F
import numpy as np
import pickle
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch_geometric import seed_everything
import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv, TransformerConv, GINEConv, EdgeConv, GeneralConv, GCNConv, GATConv, GATv2Conv
from torch.nn import Linear
import argparse
import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, edge_dim, num_layers, model_type):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        if model_type == 'GraphSAGE':
            self.conv = SAGEEdgeConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        elif model_type == 'GraphTransformer':
            self.conv = TransformerConv((-1, -1), hidden_channels, edge_dim=edge_dim)
        elif model_type == 'GINE':
            self.conv = GINEConv(Linear(hidden_channels, hidden_channels), edge_dim=edge_dim)
        elif model_type == 'EdgeConv':
            self.conv = EdgeConvConv(Linear(2 * hidden_channels + edge_dim, hidden_channels), train_eps=True,
                                     edge_dim=edge_dim)
        elif model_type == 'GeneralConv':
            self.conv = GeneralConv((-1, -1), hidden_channels, in_edge_channels=edge_dim)
        elif model_type == 'GCN':
            self.conv = SAGEConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv = GATConv(hidden_channels, hidden_channels, add_self_loops=False)  # 禁用自环添加
        elif model_type == 'RevGAT':
            self.conv = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim, add_self_loops=False)
        elif model_type == 'MaxSAGE':
            self.conv = SAGEConv(hidden_channels, hidden_channels, aggr='max')
        elif model_type == 'MeanSAGE':
            self.conv = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        elif model_type == 'GIN':
            self.conv = GINConv(Linear(hidden_channels, hidden_channels))
        else:
            raise NotImplementedError('Model type not implemented')

        for _ in range(num_layers):
            self.convs.append(self.conv)
        self.model_type = model_type
        
    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs):
            if self.model_type == "GCN":
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
            x = x.relu() if i != len(self.convs) - 1 else x
        return x



class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 4)
        self.lin2 = Linear(hidden_channels // 4, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_dim, num_layers, model_type):
        super().__init__()
        self.model_type = model_type
        if model_type != 'MLP':
            self.gnn = GNN(hidden_channels, edge_dim, num_layers, model_type=model_type)

        self.classifier = Classifier(hidden_channels, out_channels)

    def forward(self, data):
        x = data.x
        if self.model_type != 'MLP':
            x = self.gnn(x, data.edge_index, data.edge_attr)

        pred = self.classifier(x)
        return pred


if __name__ == '__main__':
    seed_everything(66)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', '-dt', type=str, default='children', help='Data type')
    parser.add_argument('--emb_type', '-et', type=str, default='openai-new', help='Embedding type')  # TODO: set edge dim
    parser.add_argument('--model_type', '-mt', type=str, default='GraphTransformer', help='Model type')
    args = parser.parse_args()

    # Dataset = Children(root='.') 
    # data = Dataset[0]   # TODO: Citation code in TAG
    if args.data_type == 'app':
        with open(f'dataset/raw/{args.data_type}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif args.data_type == 'movie':
        with open(f'dataset/raw/{args.data_type}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif args.data_type == 'children':
        with open(f'dataset/raw/{args.data_type}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif args.data_type == 'history':
        with open(f'dataset/raw/{args.data_type}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif args.data_type == 'mystery':
        with open(f'dataset/raw/{args.data_type}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif args.data_type == 'comics':
        with open(f'dataset/raw/{args.data_type}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif args.data_type == 'reddit':
        with open(f'dataset/raw/{args.data_type}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif args.data_type == 'twitter':
        with open(f'dataset/raw/{args.data_type}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif args.data_type == 'citation':
        with open(f'dataset/raw/{args.data_type}.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        raise NotImplementedError('Dataset not implemented')

    num_nodes = len(data.text_nodes)
    num_edges = len(data.text_edges)

    #print(data.text_node_labels)
    product_indices = torch.tensor([i for i, label in enumerate(data.text_node_labels) if label != -1])
    product_labels = [label for label in data.text_node_labels if label != -1]
    real_id = 0
    n_id_to_index = {}
    for i, label in enumerate(data.text_node_labels):
        if label != -1:
            n_id_to_index[i] = real_id
            real_id += 1


    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(product_labels) 
    data.y = torch.tensor(binary_labels).long()#.float()

    train_ratio = 0.8
    val_ratio = 0.1

    num_train_products = int(len(product_labels) * train_ratio)
    num_val_products = int(len(product_labels) * val_ratio)
    num_test_products = int(len(product_labels)) - num_train_products - num_val_products#######
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[product_indices[:num_train_products]] = 1
    data.val_mask[product_indices[num_train_products:num_train_products + num_val_products]] = 1
    data.test_mask[product_indices[-num_test_products:]] = 1

    data.num_classes = len(mlb.classes_)  
    #data.num_nodes = num_nodes

    del data.text_nodes
    del data.text_node_labels
    del data.text_edges
    del data.edge_score

    # set hidden channels and edge dim for diff emb type 

    if args.emb_type != 'None':
        data.x = torch.randn(num_nodes, 128).squeeze().float()#torch.load(f'dataset/emb/{args.data_type}_{args.emb_type}_node.pt').squeeze().float()
        data.edge_attr = torch.randn(num_edges, 128).squeeze().float()#torch.load(f'dataset/emb/{args.data_type}_openai_edge.pt').squeeze().float()
        if args.emb_type == 'openai-new':
            edge_dim = 128
            node_dim = 128
        elif args.emb_type == 'openai-old':
            edge_dim = 128
            node_dim = 128
        elif args.emb_type == 'Large_Bert':
            edge_dim = 1024
            node_dim = 1024
        elif args.emb_type == 'Angle':
            edge_dim = 1024
            node_dim = 1024
        else:
            raise NotImplementedError('Embedding type not implemented')
    else:
        data.x = torch.load(f'citation_dataset/emb/citation_Large_Bert_node.pt').squeeze().float()
        data.edge_attr = torch.randn(num_edges, 128).squeeze().float()
        edge_dim = 128
        node_dim = 128

    # Make sure all attributes of data are contiguous
    data.x = data.x.contiguous()
    data.edge_index = data.edge_index.contiguous()
    
    print(data)

    train_loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[10,10], batch_size=512,
                                  shuffle=True)
    val_loader = NeighborLoader(data, input_nodes=data.val_mask, num_neighbors=[10, 10], batch_size=1024, shuffle=False)
    test_loader = NeighborLoader(data, input_nodes=data.test_mask, num_neighbors=[10, 10], batch_size=1024,
                                 shuffle=False)

    train_loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[10, 10], batch_size=1024,
                                  shuffle=True)
    val_loader = NeighborLoader(data, input_nodes=data.val_mask, num_neighbors=[10, 10], batch_size=1024, shuffle=False)
    test_loader = NeighborLoader(data, input_nodes=data.test_mask, num_neighbors=[10, 10], batch_size=1024,
                                 shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = Model(hidden_channels=node_dim, out_channels=data.num_classes, edge_dim=edge_dim, num_layers=2,
                  model_type=args.model_type)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    if args.data_type in ["twitter", "reddit", "citation"]:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    for epoch in range(1, 10):
        model.train()
        total_examples = total_loss = 0

        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            batch_size = batch.batch_size
            
            out = model(batch)
            new_indices = [n_id_to_index.get(n_id.item(), -1) for n_id in batch.n_id]
            valid_indices = [index for index in new_indices if index != -1]
            judge = [True if x != -1 else False for x in new_indices]
            if valid_indices:  
                true = batch.y[torch.tensor(valid_indices)].to(device)
                out_filtered = out[torch.tensor(judge, dtype=torch.bool)]
            else:
                raise ValueError("No valid indices found for batch.y.")

            loss = criterion(out_filtered, true.float())
            loss.backward()
            optimizer.step()

            total_examples += batch_size
            total_loss += float(loss) * batch_size

        if epoch % 1 == 0 and epoch != 0:
            print('Validation begins')

            model.eval()
            with torch.no_grad():
                preds = []
                ground_truths = []
                for batch in tqdm.tqdm(val_loader):
                    batch = batch.to(device)
    
                    out = model(batch)
                    pred = F.softmax(out, dim=1) 
                    new_indices = [n_id_to_index.get(n_id.item(), -1) for n_id in batch.n_id]
                    valid_indices = [index for index in new_indices if index != -1]
                    judge = [True if x != -1 else False for x in new_indices]
                    if valid_indices:  
                        true = batch.y[torch.tensor(valid_indices)].to(device)
                        out_filtered = out[torch.tensor(judge, dtype=torch.bool)]
                    
                    preds.append(out_filtered)
                    ground_truths.append(true)

                pred = torch.cat(preds, dim=0).cpu().numpy()
                ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
                
                y_pred_labels = (pred > 0.5).astype(int) 

                f1 = f1_score(ground_truth, y_pred_labels, average='samples')
                print(f"F1 score: {f1:.4f}")
                
                if args.data_type not in ["twitter", "reddit", "citation"]:
                    micro_auc = roc_auc_score(ground_truth, pred, average='micro')
                    print(f"Validation micro AUC: {micro_auc:.4f}")
                
                # ACC
                accuracy = accuracy_score(ground_truth, y_pred_labels)
                print(f"Validation Accuracy: {accuracy:.4f}")
              
                  