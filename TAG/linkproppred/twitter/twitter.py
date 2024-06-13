import os
import pandas as pd
import numpy as np
import pickle as pkl
import networkx as nx
import json
import os.path as osp
from typing import List

import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm


from torch_geometric.data import Data
import torch
from torch_geometric.io import fs, read_planetoid_data


class Twitter(InMemoryDataset):
	url =  "https://github.com/YuweiCao-UIC/KPGNN/raw/main/datasets/Twitter"
	
	def __init__(self, root: str) -> None:
		super().__init__(root)
		self.load(self.processed_paths[0])

	@property
	def raw_dir(self) -> str:
		return osp.join(self.root, 'twitter_dataset', 'raw')

	@property
	def processed_dir(self) -> str:
		return osp.join(self.root, 'twitter_dataset', 'processed')

	@property
	def raw_file_names(self) -> List[str]:
		file_names = [
			'68841_tweets_multiclasses_filtered_0722_part1.npy', 
			'68841_tweets_multiclasses_filtered_0722_part2.npy'
		]

		return file_names

	def download(self) -> None:
		for name in self.raw_file_names:
			fs.cp(f'{self.url}/{name}', self.raw_dir)
	
	@property
	def processed_file_names(self) -> str:
		return 'data.pt'
	

	def process(self) -> None:
		
		# Define file paths
		p_part1 = osp.join(self.raw_dir, '68841_tweets_multiclasses_filtered_0722_part1.npy')
		p_part2 = osp.join(self.raw_dir, '68841_tweets_multiclasses_filtered_0722_part2.npy')

		# Load the numpy data
		df_np_part1 = np.load(p_part1, allow_pickle=True)
		df_np_part2 = np.load(p_part2, allow_pickle=True)
		df_np = np.concatenate((df_np_part1, df_np_part2), axis=0)

		# Convert numpy data to pandas DataFrame
		df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
			"place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities",
			"words", "filtered_words", "sampled_words"])
		df_graph_4 = df[['tweet_id', 'text', 'user_id', 'user_mentions', 'event_id']]

		# Create the graph
		text_nodes, text_edges, node_labels, edge_labels, edge_index = self.create_graph_4(df_graph_4)

		# Create Data object
		graph = Data(
			text_nodes=text_nodes,
			text_edges=text_edges,
			node_labels=torch.tensor(node_labels, dtype=torch.long),
			edge_labels=torch.tensor(edge_labels, dtype=torch.long),
			edge_index=torch.tensor(edge_index, dtype=torch.long)
		)
		
		self.save([graph], self.processed_paths[0])


	def create_graph_4(self, df_graph):
		# Create an empty graph
		G = nx.Graph()
		text_nodes = []
		edge_index = [[], []]
		text_edges = []
		node_labels = []
		edge_labels = []

		# Create all user nodes and user_id2idx
		user_id2idx = {}
		user_nodes = []
		for _, row in df_graph.iterrows():
			mentions_list = row['user_mentions']
			user_id = row["user_id"]
			# Only include user nodes where users mention each other
			if len(mentions_list) > 0:
				if user_id not in user_id2idx:
					user_id2idx[user_id] = len(user_nodes)
					user_nodes.append({user_id: user_id2idx[user_id]})
					G.add_node(user_id2idx[user_id])
				for mention in mentions_list:
					if mention not in user_id2idx:
						user_id2idx[mention] = len(user_nodes)
						user_nodes.append({mention: user_id2idx[mention]})
						G.add_node(user_id2idx[mention])

		text_nodes = ["user"] * len(user_nodes)  # All user nodes have text "user" and label -1
		node_labels = [-1] * len(user_nodes)
		print("Length of user nodes:", len(user_nodes))
		print("Sample user nodes:", user_nodes[:5])
		print("Sample user node labels:", node_labels[:5])
		print("Sample user node texts:", text_nodes[:5])

		# Initialize tweet_id2idx
		tweet_id2idx = {}
		tweet_id2node_idx = {}
		tweet_nodes = []

		# Add information related to mentions (user-user edge, tweet node, user-tweet edge)
		for _, row in df_graph.iterrows():
			mentions_list = row['user_mentions']
			if len(mentions_list) > 0:
				user_idx = user_id2idx[row['user_id']]

				for mention in mentions_list:
					mention_idx = user_id2idx[mention]
					if not G.has_edge(user_idx, mention_idx):  # Only include one edge between two users
						# Add edge u1-u2
						G.add_edge(user_idx, mention_idx)
						edge_index[0].append(user_idx)
						edge_index[1].append(mention_idx)
						text_edges.append(row['text'])
						edge_labels.append(row["event_id"])

						# Add node t1, create tweet_id2idx
						tweet_id = row['tweet_id']
						tweet_id2idx[tweet_id] = len(tweet_nodes)
						tweet_id2node_idx[tweet_id] = len(user_nodes) + tweet_id2idx[tweet_id]
						tweet_nodes.append({tweet_id: len(user_nodes) + tweet_id2idx[tweet_id]})
						G.add_node(tweet_id2node_idx[tweet_id])
						text_nodes.append(row["text"])
						node_labels.append(row['event_id'])

						# Add edge u1-t1
						tweet_node_idx = tweet_id2node_idx[tweet_id]
						edge_index[0].append(user_idx)
						edge_index[1].append(tweet_node_idx)
						text_edges.append("")
						edge_labels.append(-1)
						G.add_edge(user_idx, tweet_node_idx)

						# Add edge t1-u2
						edge_index[0].append(mention_idx)
						edge_index[1].append(tweet_node_idx)
						text_edges.append("")
						edge_labels.append(-1)
						G.add_edge(mention_idx, tweet_node_idx)

		# Add information not related to mentions (user-tweet edge, tweet node)
		for _, row in df_graph.iterrows():
			tweet_id = row['tweet_id']
			user_id = row["user_id"]
			if user_id in user_id2idx.keys() and tweet_id not in tweet_id2idx.keys():
				# Create tweet_id2node_idx and add node t1
				tweet_id2idx[tweet_id] = len(tweet_nodes)
				tweet_id2node_idx[tweet_id] = len(user_nodes) + tweet_id2idx[tweet_id]
				tweet_nodes.append({tweet_id: len(user_nodes) + tweet_id2idx[tweet_id]})
				G.add_node(tweet_id2node_idx[tweet_id])
				text_nodes.append(row["text"])
				node_labels.append(row['event_id'])

				# Add edge u1-t1
				user_idx = user_id2idx[user_id]
				tweet_node_idx = tweet_id2node_idx[tweet_id]
				edge_index[0].append(user_idx)
				edge_index[1].append(tweet_node_idx)
				text_edges.append("")
				edge_labels.append(-1)
				G.add_edge(user_idx, tweet_node_idx)

			print("Length of tweet nodes:", len(tweet_nodes))
			print("Sample tweet nodes:", tweet_nodes[:5])
			print("Sample tweet node labels:", node_labels[-5:])
			print("Sample tweet node texts:", text_nodes[-5:])

			return text_nodes, text_edges, node_labels, edge_labels, edge_index

if __name__ == "__main__":
	dataset = Twitter(root=".")
	print(dataset[0])
