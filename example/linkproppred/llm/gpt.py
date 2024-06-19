import sys
import os

from torch_geometric import seed_everything

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pickle
import openai
import random
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from TAG.linkproppred.reddit import Reddit
from TAG.linkproppred.citation import citation
from TAG.linkproppred.twitter import Twitter


def truncate_text(text, max_tokens):
    tokens = text.split()
    return ' '.join(tokens[:max_tokens]) if len(tokens) > max_tokens else text


def extract_numbers(text):
    return text.split(",")[0]


if __name__ == "__main__":
    seed_everything(66)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', '-dt', type=str, default='twitter',
                        help='data type for datasets, options are reddit, citation, twitter')
    parser.add_argument('--GPT_type', '-gt', type=str, default='gpt-3.5-turbo',
                        help='LLM type, options are gpt-3.5-turbo, gpt-4')
    parser.add_argument('--predict_node_number', '-n', type=int, default=1000,
                        help='Number of ')
    args = parser.parse_args()
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_type)

    if args.data_type == 'reddit':
        Dataset = Reddit(root=dataset_path)
        system_prompt = '''
            This is the Reddit dataset, you need to do link prediction based on the nodes' text that I give to you. The data is sourced from Reddit, represent two prominent social media platforms.Nodes represent users and topics. 
            Below is a list of reddit's node pairs information, which means each information is a description of the node. Your task is to predict whether there should be a link between these two nodes.
            This is the rules for your link prediction:
            0:no link
            1:has link
            Attention, you only need to reply with the relevant code and its label name seperated by a comma, about whether there is a link between the nodes's pairs information that I give to you. You must give me an answer based on the format that I give to you below.
            Below is an reply example of link prediciton:
            Example: 0,no link
                     1,has link
                     1, has link
            '''
    elif args.data_type == 'citation':
        Dataset = Citation(root=dataset_path)
        system_prompt = '''
            The raw data for the citation network is sourced from the Open Research Corpus, derived from the complete Semantic Scholar 
                    corpus. Your task is to do link prediction. Nodes represent papers and edges represent the citation relationships 
                    between papers. The descriptions of papers are used as node textual information, and citation information, 
                    such as the context and paragraphs in which papers are cited, is utilized as textual edge data.
                    This is the rules for your link prediction:
                    0:no link
                    1:has link
                    Attention, you only need to reply with the relevant code and its label name seperated by a comma, about whether there is a link between the nodes's pairs information that I give to you. You must give me an answer based on the format that I give to you below.
                    Below is an reply example of link prediciton:
                    Example: 0,no link
                             1,has link
                             1, has link
                    '''
    if args.data_type == 'twitter':
        Dataset = Twitter(root=dataset_path)
        system_prompt = '''
                    Below is a twitter dataset, you need to do link prediction based on the nodes' text that I give to you. Nodes represent users and nodes topics. The edges in the dataset indicate two types of reviews: those between users, representing.
                    user-user links, and those between users and topics, representing user-topic links. The descriptions of topics are used as node textual information, and reviews are used as edge textual information.
                    Attention, reply with the relevant code, separated by a comma. 
                    This is the rules for your link prediction:
                    0:no link
                    1:has link
                    The most important is, you only need to reply the relevant code, separated by a comma. Do not let text or some other strange symbols exist.
                    Below is an reply example:
                    Example: 0,no link
                             1,has link
                             1, has link
                    '''
    else:
        raise NotImplementedError('Dataset not implemented')

    data = Dataset[0]

    num_nodes = len(data.text_nodes)
    num_nodes_for_predict = args.predict_node_number

    has_text_index1 = [int(x) for x in data.edge_index[0] if
                       data.text_nodes[x] != '']  # TODO: node numbers are not aligin
    has_text_index2 = [int(x) for x in data.edge_index[1] if data.text_nodes[x] != '']

    record = [i for i in range(num_nodes) if
              data.text_nodes[data.edge_index[0][i]] != '' and data.text_nodes[data.edge_index[1][i]] != '']

    random_numbers1 = random.sample(has_text_index1, int(num_nodes_for_predict / 2))
    random_numbers2 = random.sample(has_text_index2, int(num_nodes_for_predict / 2))
    node_pairs_index = [(random_numbers1[i], random_numbers2[i]) for i in range(int(num_nodes_for_predict / 2))]

    for i in record[:int(num_nodes_for_predict / 2)]:
        node_pairs_index.append((int(data.edge_index[0][i]), int(
            data.edge_index[1][i])))  # the record is used to ensure that we include some nodes that have links

    actual_labels = []
    edge_index = np.array(data.edge_index)
    for pairs in node_pairs_index:
        edge_exists = np.any((edge_index[0] == pairs[0]) & (edge_index[1] == pairs[1]))
        actual_labels.append(1 if edge_exists else 0)

    client = openai.OpenAI(api_key="")  # TODO: add your openai api key

    max_tokens_per_request = 4000
    predicted_list = []
    input_list = []

    node_pairs = [(data.text_nodes[x[0]], data.text_nodes[x[1]]) for x in node_pairs_index]

    print(f"Current Dataset {args.data_type}. Loaded {len(node_pairs)} nodes")
    for i in tqdm(range(len(node_pairs))):
        node1_text = truncate_text(node_pairs[i][0], max_tokens_per_request // 2 - len(system_prompt.split()))
        node2_text = truncate_text(node_pairs[i][1], max_tokens_per_request // 2 - len(system_prompt.split()))
        truncated_node_text = node1_text + "\n" + node2_text

        completion_res = client.chat.completions.create(
            model=f"{args.GPT_type}",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": truncated_node_text}
            ]
        )
        input_list.append(truncated_node_text)
        gpt_result = completion_res.choices[0].message.content.strip()
        predicted_list.append(gpt_result)

    pred = []
    index = []
    for i, text in enumerate(predicted_list):
        try:
            pred.append(int(extract_numbers(text)))
            index.append(i)
        except:
            continue

    actual = [actual_labels[i] for i in index]

    pred_labels_bin = pred
    true_labels_bin = actual

    f1 = f1_score(true_labels_bin, pred_labels_bin, average='weighted')
    print(f"F1 score: {f1:.4f}")

    true_labels_flat = true_labels_bin
    pred_labels_flat = pred_labels_bin
    micro_accuracy = accuracy_score(true_labels_flat, pred_labels_flat)
    print(f"Validation micro ACC : {micro_accuracy:.4f}")
