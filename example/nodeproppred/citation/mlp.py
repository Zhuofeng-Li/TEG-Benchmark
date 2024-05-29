import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch.nn.functional as F
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv, TransformerConv, GINEConv, EdgeConv, GeneralConv
from torch.nn import Linear
import argparse


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
		self.classifier = Classifier(hidden_channels, out_channels)

	def forward(self, data):
		x = data.x
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


	num_nodes = len(data.text_nodes)
	num_edges = len(data.text_edges)

	# map node labels 
	label_to_int = {label: i for i, label in enumerate(set(data.text_node_labels))}
	data.y = torch.tensor([label_to_int[label] for label in data.text_node_labels]).long()
 
	# data split
	train_ratio = 0.8
	val_ratio = 0.1

	num_train_paper = int(num_nodes * train_ratio)
	num_val_paper = int(num_nodes * val_ratio)
	num_test_paper = num_nodes - num_train_paper - num_val_paper
			
	paper_indices = torch.randperm(num_nodes)

	data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
	data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
	data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

	data.train_mask[paper_indices[:num_val_paper]] = 1
	data.val_mask[paper_indices[num_val_paper:num_val_paper + num_val_paper ]] = 1
	data.test_mask[paper_indices[-num_test_paper:]] = 1

	data.num_classes = max(data.y) + 1
 
	del data.text_nodes
	del data.text_node_labels
	del data.text_edges

	# load emb
	if args.emb_type == 'Angle':  # TODO: reset emb name
		data.edge_attr = torch.load('citation_dataset/emb/angle-edge.pt').squeeze().float()
		data.x = torch.load('citation_dataset/emb/angle-node.pt').squeeze().float()
	elif args.emb_type == 'None':
		data.edge_attr = torch.randn(num_edges, 1024).squeeze().float()
		data.x = torch.load('citation_dataset/emb/angle-node.pt').squeeze().float()
	
	train_loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[10, 10], batch_size=1024, shuffle=True)
	val_loader = NeighborLoader(data, input_nodes=data.val_mask, num_neighbors=[10, 10], batch_size=1024, shuffle=False)
	test_loader = NeighborLoader(data, input_nodes=data.test_mask, num_neighbors=[10, 10], batch_size=1024, shuffle=False)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	model = Model(hidden_channels=1024, out_channels=data.num_classes, edge_dim=1024, num_layers=2, model_type=args.model_type)
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	criterion = torch.nn.CrossEntropyLoss()

	for epoch in range(1, 10):
		model.train()
		total_examples = total_loss = 0

		for batch in tqdm.tqdm(train_loader):
			optimizer.zero_grad()
			batch = batch.to(device)
			batch_size = batch.batch_size
			
			out = model(batch)
			loss = criterion(out, batch.y)
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
					
					preds.append(pred)
					ground_truths.append(batch.y)
				
				pred = torch.cat(preds, dim=0).cpu().numpy()
				ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
				
				# F1 Score
				y_pred_labels = np.argmax(pred, axis=1)  # 获得预测类别
				f1 = f1_score(ground_truth, y_pred_labels, average='weighted')
				print(f"F1 score: {f1:.4f}")
				
				# ACC
				accuracy = accuracy_score(ground_truth, y_pred_labels)
				print(f"Validation Accuracy: {accuracy:.4f}")
