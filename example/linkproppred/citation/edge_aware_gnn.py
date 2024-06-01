import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, TransformerConv, GINEConv, GeneralConv, EdgeConv
from torch.nn import Linear
from models import SAGEEdgeConv, EdgeConvConv, MLP
import argparse

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
		else:
			raise NotImplementedError('Model type not implemented')

		for _ in range(num_layers):
			self.convs.append(self.conv)

	def forward(self, x, edge_index, edge_attr):
		for i, conv in enumerate(self.convs):
			x = conv(x, edge_index, edge_attr)
			x = x.relu() if i != len(self.convs) - 1 else x
		return x

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

class Model(torch.nn.Module):
	def __init__(self, hidden_channels, edge_dim, num_layers, model_type):
		super().__init__()
		self.model_type = model_type
		if model_type != 'MLP':
			self.gnn = GNN(hidden_channels, edge_dim, num_layers, model_type=model_type)
		
		self.classifier = Classifier(hidden_channels)

	def forward(self, data):
		x = data.x
		if self.model_type != 'MLP':
			x = self.gnn(x, data.edge_index, data.edge_attr)
			
		pred = self.classifier(x, data.edge_label_index)
		return pred, x


if __name__ == "__main__":
	seed_everything(66)

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_type', '-dt', type=str, default='citation', help='Data type')
	parser.add_argument('--emb_type', '-et', type=str, default='Angle', help='Embedding type')  # TODO: set edge dim
	parser.add_argument('--model_type', '-mt', type=str, default='GraphTransformer', help='Model type')
	args = parser.parse_args()

	# Dataset = Children(root='.') 
	# data = Dataset[0]   # TODO: Citation code in TAG
	with open(f'citation_dataset/raw/{args.data_type}.pkl', 'rb') as f:
		data = pickle.load(f)
	
	num_nodes = len(data.text_nodes)
	num_edges = len(data.text_edges)
 
	del data.text_nodes
	del data.text_node_labels
	del data.text_edges

	# set hidden channels and edge dim for diff emb type 
	if args.emb_type != 'None':
		data.x = torch.load(f'citation_dataset/emb/{args.data_type}_{args.emb_type}_node.pt').squeeze().float()
		data.edge_attr = torch.load(f'citation_dataset/emb/{args.data_type}_{args.emb_type}_edge.pt').squeeze().float()
		if args.emb_type == 'GPT-3.5-TURBO':
			edge_dim = 1536
			node_dim = 1536
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
		data.edge_attr = torch.randn(num_edges, 1024).squeeze().float()
		edge_dim = 1024
		node_dim = 1024

	print(data)
  
	train_data, val_data, test_data = T.RandomLinkSplit(
		num_val=0.8,
		num_test=0.1,
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
		batch_size=1024,
		shuffle=True,
	)

	edge_label_index = val_data.edge_label_index
	edge_label = val_data.edge_label
	val_loader = LinkNeighborLoader(
		data=val_data,
		num_neighbors=[20, 10],
		edge_label_index=(edge_label_index),
		edge_label=edge_label,
		batch_size=1024,
		shuffle=False,
	)

	edge_label_index = test_data.edge_label_index
	edge_label = test_data.edge_label
	test_loader = LinkNeighborLoader(
		data=test_data,
		num_neighbors=[20, 10],
		edge_label_index=(edge_label_index),
		edge_label=edge_label,
		batch_size=1024,
		shuffle=False,
	)
	
	model = Model(hidden_channels=node_dim, edge_dim=edge_dim, num_layers=2, model_type=args.model_type)  # TODO: edge dim
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	model = model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	for epoch in range(1, 10):
		total_loss = total_examples = 0
		for sampled_data in tqdm.tqdm(train_loader):
			optimizer.zero_grad()
			sampled_data = sampled_data.to(device)
			pred, x = model(sampled_data)
			ground_truth = sampled_data.edge_label  
			loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
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
						pred = model(sampled_data)[0]
						preds.append(pred)
						ground_truths.append(sampled_data.edge_label)
						positive_pred = pred[sampled_data.edge_label == 1].cpu().numpy()
						negative_pred = pred[sampled_data.edge_label == 0].cpu().numpy()
					pred = torch.cat(preds, dim=0).cpu().numpy()

				ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
				y_label = np.where(pred >= 0.5, 1, 0)
				f1 = f1_score(ground_truth, y_label)
				print(f"F1 score: {f1:.4f}")
				# AUC
				auc = roc_auc_score(ground_truth, pred)
				print(f"Validation AUC: {auc:.4f}")
