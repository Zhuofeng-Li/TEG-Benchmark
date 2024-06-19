import pickle
import openai
import random
from openai import OpenAI
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer


# from TAG.linkproppred.reddit import Reddit
# from TAG.linkproppred.citation import Citation

def truncate_text(text, max_tokens):
    tokens = text.split()
    return ' '.join(tokens[:max_tokens]) if len(tokens) > max_tokens else text


def extract_numbers(text):
    return text.split(",")[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', '-dt', type=str, default='reddit',
                        help='data type for datasets, options are reddit, citation, twitter')
    parser.add_argument('--GPT_type', '-gt', type=str, default='gpt-3.5-turbo',
                        help='LLM type, options are gpt-3.5-turbo, gpt-4')
    parser.add_argument('--predict_node_number', '-n', type=int, default=1000,
                        help='Number of ')
    args = parser.parse_args()

    if args.data_type == 'reddit':
        # Dataset = Reddit(root=f'{args.data_type}')
        labels_dict = {0: "user nodes,not a moderator for Reddit community",
                       1: "user nodes,a moderator for Reddit community",
                       -1: "Subreddit nodes"}
        system_prompt = '''
            The Reddit dataset, sourced from Reddit, represent two prominent social media platforms.Nodes represent users and
            204 topics. 
            Below is a list of reddit label classifications, which means which classification this nodes belong to and a sample context which is the description of this node is given to you to help you judge.
            Based on the context provided, determine the label classification code for the node described. 
            Attention, reply with the relevant code and its label name, separated by a comma. 
            Below is an reply example:
            Example: 0,user nodes,not a moderator for Reddit community
            '''
    elif args.data_type == 'citation':
        Dataset = Citation(root=f'{args.data_type}')
        system_prompt = '''
                    The raw data for the citation network is sourced from the Open Research Corpus, derived from the complete Semantic Scholar 
                    corpus. Your task is to do node classification. Nodes represent papers. The descriptions of papers are used as node textual 
                    information
                    Based on the description provided, determine the label classification code for the node described. 
                    Attention, reply with the relevant code and its label name, separated by a comma. 
                    Below is an reply example:
                    Example: 0,
                    '''
    else:
        raise NotImplementedError('Dataset not implemented')

    # data = Dataset[0]
    with open("reddit_graph.pkl", "rb") as f:
        data = pickle.load(f)
    if args.data_type == 'citation':
        labels_dict = {i: value for i, value in enumerate(set(data.text_nodes))}

    category_to_label = {category: i for i, category in enumerate(labels_dict)}

    labels = "\nLabels Type Codes:\n"
    for key, value in labels_dict.items():
        labels += f"{value}: {key}\n"
    print(labels)

    client = openai.OpenAI(api_key="your api key")

    system_prompt = system_prompt + labels

    max_tokens_per_request = 4000

    predicted_list = []
    input_list = []

    num_nodes = args.predict_node_number

    node_index = [random.randint(0, len(data.text_nodes) - 1) for _ in range(num_nodes)]
    print(f"{num_nodes} nodes loaded")
    for i in tqdm(range(num_nodes)):
        truncated_node_text = truncate_text(data.text_nodes[node_index[i]],
                                            max_tokens_per_request - len(system_prompt.split()))

        completion_res = client.chat_completions.create(
            model=f"{args.GPT_type}",  # here model is a string of openai model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": truncated_node_text}
            ]
        )
        input_list.append(truncated_node_text)
        gpt_result = completion_res.choices[0].message.content  # here ans should be the string reply
        predicted_list.append(gpt_result)

    pred = []
    index = []
    for i, text in enumerate(predicted_list):
        try:
            pred.append(int(extract_numbers(text)))
            index.append(i)
        except:
            continue

    actual_category = [data.node_labels[x] for x in node_index]
    actual = [actual_category[i] for i in index]

    mlb = MultiLabelBinarizer()
    pred_labels_bin = pred
    true_labels_bin = actual

    f1 = f1_score(true_labels_bin, pred_labels_bin, average='weighted')
    print(f"F1 score: {f1:.4f}")

    true_labels_flat = true_labels_bin
    pred_labels_flat = pred_labels_bin
    micro_accuracy = accuracy_score(true_labels_flat, pred_labels_flat)
    print(f"Validation micro ACC : {micro_accuracy:.4f}")
