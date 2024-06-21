import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from models import SAGEEdgeConv, EdgeConvConv, MLP
import torch.nn.functional as F
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch_geometric import seed_everything
import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv, TransformerConv, GINEConv, EdgeConv, GeneralConv
from torch.nn import Linear
import argparse

from TAG.linkproppred.twitter import Twitter
from TAG.linkproppred.reddit import Reddit
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
			x = self.gnn(x, data.edge_index, edge_attr=data.edge_attr)
		
		pred = self.classifier(x)
		return pred

if __name__ == '__main__':
	seed_everything(66)

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_type', '-dt', type=str, default='reddit', help='Data type:twitter, reddit')
	parser.add_argument('--emb_type', '-et', type=str, default='None', help='Embedding type:GPT-3.5-TURBO, BERT, Bert_Large, None')  # TODO: set edge dim
	parser.add_argument('--model_type', '-mt', type=str, default='GraphSAGE', help='Model type:MLP, GraphSAGE, EdgeConv, GeneralConv, GraphTransformer, GINE')
	args = parser.parse_args()

	if args.data_type == 'twitter':
		Dataset = Twitter(root=f'{args.data_type}')
	elif args.data_type == 'reddit':
		Dataset = Reddit(root=f'{args.data_type}')
	else:
		raise NotImplementedError('Dataset not implemented')
	data = Dataset[0]
	print(data)

	num_nodes = len(data.text_nodes)
	num_edges = len(data.text_edges)

	# map node labels 
	node_labels=data.node_labels.tolist()
	label_to_int = {label: i for i, label in enumerate(set(node_labels))}
	data.y = torch.tensor([label_to_int[label] for label in node_labels]).long()
 
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
	data.num_nodes = num_nodes
 
	del data.text_nodes
	del data.text_node_labels
	del data.text_edges

	# set hidden channels and edge dim for diff emb type 
	
	if args.emb_type != 'None':
		data.x = torch.load(f'{args.data_type}_dataset/emb/{args.data_type}_{args.emb_type}_node.pt').squeeze().float().contiguous()
		data.edge_attr = torch.load(f'{args.data_type}_dataset/emb/{args.data_type}_{args.emb_type}_edge.pt').squeeze().float().contiguous()
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
		#data.x = torch.load(f'{args.data_type}_dataset/emb/citation_Large_Bert_node.pt').squeeze().float().contiguous()
		data.x = torch.randn(num_nodes, 1024).squeeze().float()
		data.edge_attr = torch.randn(num_edges, 1024).squeeze().float()
		edge_dim = 1024
		node_dim = 1024
	
	# Make sure all attributes of data are contiguous
	data.x = data.x.contiguous()
	data.edge_index = data.edge_index.contiguous()
	
	print(data)	

	# Now create the NeighborLoaders
	train_loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[10, 10], batch_size=1024, shuffle=True)
	val_loader = NeighborLoader(data, input_nodes=data.val_mask, num_neighbors=[10, 10], batch_size=1024, shuffle=False)
	test_loader = NeighborLoader(data, input_nodes=data.test_mask, num_neighbors=[10, 10], batch_size=1024, shuffle=False)

	train_loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[10, 10], batch_size=1024, shuffle=True)
	val_loader = NeighborLoader(data, input_nodes=data.val_mask, num_neighbors=[10, 10], batch_size=1024, shuffle=False)
	test_loader = NeighborLoader(data, input_nodes=data.test_mask, num_neighbors=[10, 10], batch_size=1024, shuffle=False)
 
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	model = Model(hidden_channels=node_dim, out_channels=data.num_classes, edge_dim=edge_dim, num_layers=2, model_type=args.model_type)
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
