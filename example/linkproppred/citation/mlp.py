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
import argparse


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
		self.classifier = Classifier(hidden_channels)

	def forward(self, data):
		pred = self.classifier(data.x, data.edge_label_index)
		return pred, None




if __name__ == "__main__":
	seed_everything(66)

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_type', '-dt', type=str, default='children', help='Data type')
	parser.add_argument('--emb_type', '-et', type=str, default='Angle', help='Embedding type')  # TODO: set edge dim
	parser.add_argument('--model_type', '-mt', type=str, default='GraphTransformer', help='Model type')
	args = parser.parse_args("")

	# Dataset = Children(root='.') 
	# data = Dataset[0]   # TODO: Citation code in TAG
	with open('citation_dataset/raw/filtered_citation_network.pkl', 'rb') as f:
		data = pickle.load(f)
	
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
	
	model = Model(hidden_channels=1024, edge_dim=1024, num_layers=2, model_type=args.model_type)  # TODO: edge dim
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	model = model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	for epoch in range(1, 10):
		total_loss = total_examples = 0
		for sampled_data in tqdm.tqdm(train_loader):
			optimizer.zero_grad()
			sampled_data = sampled_data.to(device)
			pred, _ = model(sampled_data)
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
