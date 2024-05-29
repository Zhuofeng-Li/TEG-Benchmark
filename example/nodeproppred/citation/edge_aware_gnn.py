import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import torch
import torch_geometric.transforms as T
import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv, TransformerConv, GINEConv, EdgeConv, GeneralConv
from torch.nn import Linear
import argparse

class GNN(torch.nn.Module):
	def __init__(self, hidden_channels, edge_dim, num_layers, model_type):
		super().__init__()
		self.convs = torch.nn.ModuleList()

		if model_type == 'GraphSAGE':
			self.conv = SAGEConv((-1, -1), hidden_channels)
		elif model_type == 'GraphTransformer':
			self.conv = TransformerConv((-1, -1), hidden_channels, edge_dim=edge_dim)
		elif model_type == 'GINE':
			self.conv = GINEConv(Linear(hidden_channels, hidden_channels), edge_dim=edge_dim)
		elif model_type == 'EdgeConv':
			self.conv = EdgeConv(Linear(2 * hidden_channels + edge_dim, hidden_channels))
		elif model_type == 'GeneralConv':
			self.conv = GeneralConv((-1, -1), hidden_channels, in_edge_channels=edge_dim)
		else:
			raise NotImplementedError('Model type not implemented')

		for _ in range(num_layers):
			self.convs.append(self.conv)

	def forward(self, x, edge_index, edge_attr):
		for i, conv in enumerate(self.convs):
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
		return torch.sigmoid(x)


class Model(torch.nn.Module):
	def __init__(self, hidden_channels, out_channels, edge_dim, num_layers, model_type):
		super().__init__()
		self.embedding = torch.nn.Embedding(data.num_nodes, hidden_channels)
		self.gnn = GNN(hidden_channels, edge_dim, num_layers, model_type=model_type)
		self.classifier = Classifier(hidden_channels, out_channels)

	def forward(self, data):
		x = self.embedding(data.x)
		x = self.gnn(x, data.edge_index, data.edge_attr)
		pred = self.classifier(x)
		return pred


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_type', '-dt', type=str, default='children',
						help='goodreads dataset type for children, crime, history, mystery')
	parser.add_argument('--emb_type', '-et', type=str, default='Angle',
						help='embedding type for GNN, options are GPT-3.5-TURBO, Bert, Angle, None')
	parser.add_argument('--model_type', '-mt', type=str, default='GraphTransformer',
						help='Model type for GNN, options are GraphTransformer, GINE, Spline')
	args = parser.parse_args()

	# Dataset = Children(root='.') 
	# data = Dataset[0]   # TODO: Citation code in TAG
	with open('citation_dataset/raw/filtered_citation_network.pkl', 'rb') as f:
		data = pickle.load(f)

	print(data)

	num_nodes = len(data.text_nodes)
	num_edges = len(data.text_edges)
 
	del data.text_nodes
	del data.text_node_labels
	del data.text_edges

	# load emb
	if args.emb_type == 'Angle':  # TODO: reset emb name
		data.edge_attr = torch.load('citation_dataset/emb/angle-edge.pt').squeeze().float()
		data.x = torch.load('citation_dataset/emb/angle-node.pt').squeeze().float()
	elif args.emb_type == 'None':
		data.edge_attr = torch.randn(num_edges, 1024).squeeze().float()
		data.x = torch.tensor(num_nodes, 1024).squeeze().float()


	train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=[10, 10], batch_size=1024, shuffle=True, edge_attr=data.edge_attr)
	val_loader = NeighborSampler(data.edge_index, node_idx=data.val_mask, sizes=[10, 10], batch_size=1024, shuffle=False, edge_attr=data.edge_attr)
	test_loader = NeighborSampler(data.edge_index, node_idx=data.test_mask, sizes=[10, 10], batch_size=1024, shuffle=False, edge_attr=data.edge_attr)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	model = Model(hidden_channels=256, out_channels=data.num_classes, edge_dim=3072, num_layers=2, model_type=args.model_type)
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	weight = data.y.sum(0)
	weight = weight.max() / weight

	criterion = torch.nn.BCELoss(weight=weight)
	criterion = criterion.to(device)

	for epoch in range(1, 10):
		model.train()
		total_examples = total_loss = 0

		for batch_size, n_id, adjs in tqdm.tqdm(train_loader):
			optimizer.zero_grad()
			adjs = [adj.to(device) for adj in adjs]

			out = model(adjs[0])
			loss = criterion(out, data.y[n_id[:batch_size]].float())
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
				for batch_size, n_id, adjs in tqdm.tqdm(val_loader):
					adjs = [adj.to(device) for adj in adjs]

					pred = model(adjs[0])
					preds.append(pred)
					ground_truths.append(data.y[n_id[:batch_size]].float())

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
