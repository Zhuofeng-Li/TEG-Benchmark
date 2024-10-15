from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import numpy as np
import torch as th
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import os
import random
import pickle


def split_graph(nodes_num, train_ratio, val_ratio):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    train_size = int(nodes_num * train_ratio)
    val_size = int(nodes_num * val_ratio)

    train_ids = indices[:train_size]
    val_ids = indices[train_size : train_size + val_size]
    test_ids = indices[train_size + val_size :]

    return train_ids, val_ids, test_ids


# def split_time(g, train_year=2016, val_year=2017):
#     year = list(np.array(g.ndata['year']))
#     indices = np.arange(g.num_nodes())
#     # 1999-2014 train
#     train_ids = indices[:year.index(train_year)]
#     val_ids = indices[year.index(train_year):year.index(val_year)]
#     test_ids = indices[year.index(val_year):]
#
#     return train_ids, val_ids, test_ids


def split_time(g, train_year=2016, val_year=2017):
    np.random.seed(42)
    year = list(np.array(g.ndata["year"]))
    indices = np.arange(g.num_nodes())
    # 1999-2014 train
    # Filter out nodes with label -1
    valid_indices = [i for i in indices if g.ndata["label"][i] != -1]

    # Filter out valid indices based on years
    train_ids = [i for i in valid_indices if year[i] < train_year]
    val_ids = [i for i in valid_indices if year[i] >= train_year and year[i] < val_year]
    test_ids = [i for i in valid_indices if year[i] >= val_year]

    train_length = len(train_ids)
    val_length = len(val_ids)
    test_length = len(test_ids)

    print("Train set length:", train_length)
    print("Validation set length:", val_length)
    print("Test set length:", test_length)

    return train_ids, val_ids, test_ids


def load_data(name, train_ratio=0.6, val_ratio=0.2):
    if name == "ogbn-arxiv":
        data = DglNodePropPredDataset(name=name)
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        graph, labels = data[0]
        labels = labels[:, 0]
    elif name == "Children":
        graph = dgl.load_graphs("data/CSTAG/Children/Children.pt")[0][0]
        labels = graph.ndata["label"]
        train_idx, val_idx, test_idx = split_graph(graph.num_nodes(), 0.6, 0.2)
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    elif name == "History":
        graph = dgl.load_graphs("data/CSTAG/History/History.pt")[0][0]
        labels = graph.ndata["label"]
        train_idx, val_idx, test_idx = split_graph(graph.num_nodes(), 0.6, 0.2)
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    elif name == "Fitness":
        graph = dgl.load_graphs("data/CSTAG/Fitness/Fitness.pt")[0][0]
        labels = graph.ndata["label"]
        train_idx, val_idx, test_idx = split_graph(
            graph.num_nodes(), train_ratio, val_ratio
        )
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    elif name == "Photo":
        graph = dgl.load_graphs("data/CSTAG/Photo/Photo.pt")[0][0]
        labels = graph.ndata["label"]
        train_idx, val_idx, test_idx = split_time(graph, 2015, 2016)
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    elif name == "Computers":
        graph = dgl.load_graphs("data/CSTAG/Computers/Computers.pt")[0][0]
        labels = graph.ndata["label"]
        train_idx, val_idx, test_idx = split_time(graph, 2017, 2018)
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    elif name == "webkb-cornell":
        graph = dgl.load_graphs("data/webkb/Cornell/Cornell.pt")[0][0]
        labels = graph.ndata["label"]
        train_idx, val_idx, test_idx = split_graph(
            graph.num_nodes(), train_ratio, val_ratio
        )
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    elif name == "webkb-texas":
        graph = dgl.load_graphs("data/webkb/Texas/Texas.pt")[0][0]
        labels = graph.ndata["label"]
        train_idx, val_idx, test_idx = split_graph(
            graph.num_nodes(), train_ratio, val_ratio
        )
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    elif name == "webkb-washington":
        graph = dgl.load_graphs("data/webkb/Washington/Washington.pt")[0][0]
        labels = graph.ndata["label"]
        train_idx, val_idx, test_idx = split_graph(
            graph.num_nodes(), train_ratio, val_ratio
        )
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    elif name == "webkb-wisconsin":
        graph = dgl.load_graphs("data/webkb/Wisconsin/Wisconsin.pt")[0][0]
        labels = graph.ndata["label"]
        train_idx, val_idx, test_idx = split_graph(
            graph.num_nodes(), train_ratio, val_ratio
        )
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    else:
        raise ValueError("Not implemetned")
    return graph, labels, train_idx, val_idx, test_idx


def from_dgl(g):
    import dgl

    from torch_geometric.data import Data, HeteroData

    if not isinstance(g, dgl.DGLGraph):
        raise ValueError(f"Invalid data type (got '{type(g)}')")

    if g.is_homogeneous:
        data = Data()
        data.edge_index = th.stack(g.edges(), dim=0)

        for attr, value in g.ndata.items():
            data[attr] = value
        for attr, value in g.edata.items():
            data[attr] = value

        return data

    data = HeteroData()  # TODO: Delete this

    for node_type in g.ntypes:
        for attr, value in g.nodes[node_type].data.items():
            data[node_type][attr] = value

    for edge_type in g.canonical_etypes:
        row, col = g.edges(form="uv", etype=edge_type)
        data[edge_type].edge_index = th.stack([row, col], dim=0)
        for attr, value in g.edge_attr_schemes(edge_type).items():
            data[edge_type][attr] = value

    return data


def split_edge(
    dgl_graph,
    test_ratio=0.2,
    val_ratio=0.1,
    random_seed=42,
    neg_len="1000",
    path=None,
    way="random",
):
    if os.path.exists(os.path.join(path, f"{neg_len}/edge_split_hitsk.pt")):
        edge_split = th.load(os.path.join(path, f"{neg_len}/edge_split_hitsk.pt"))

    else:
        np.random.seed(random_seed)
        th.manual_seed(random_seed)

        graph = dgl_graph

        eids = np.arange(graph.num_edges)
        eids = np.random.permutation(eids)

        u, v = graph.edge_index

        test_size = int(len(eids) * test_ratio)
        val_size = int(len(eids) * val_ratio)

        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        val_pos_u, val_pos_v = (
            u[eids[test_size : test_size + val_size]],
            v[eids[test_size : test_size + val_size]],
        )
        train_pos_u, train_pos_v = (
            u[eids[test_size + val_size :]],
            v[eids[test_size + val_size :]],
        )

        train_edge_index = th.stack((train_pos_u, train_pos_v), dim=1)
        val_edge_index = th.stack((val_pos_u, val_pos_v), dim=1)
        test_edge_index = th.stack((test_pos_u, test_pos_v), dim=1)

        valid_neg_edge_index = th.randint(
            0, graph.num_nodes, [int(neg_len), 2], dtype=th.long
        )  # TODO: filter some edges in graph
        test_neg_edge_index = th.randint(
            0, graph.num_nodes, [int(neg_len), 2], dtype=th.long
        )

        edge_split = {
            "train": {
                "edge": train_edge_index,
                "train_edge_feature_index": eids[test_size + val_size :],
            },
            "valid": {"edge": val_edge_index, "edge_neg": valid_neg_edge_index},
            "test": {"edge": test_edge_index, "edge_neg": test_neg_edge_index},
        }

        th.save(edge_split, os.path.join(path, f"{neg_len}/edge_split_hitsk.pt"))

    return edge_split


def split_edge_mrr(
    dgl_graph,
    test_ratio=0.2,
    val_ratio=0.1,
    random_seed=42,
    neg_len="1000",
    path=None,
    way="random",
):
    if os.path.exists(os.path.join(path, f"{neg_len}/edge_split_mrr.pt")):
        edge_split = th.load(os.path.join(path, f"{neg_len}/edge_split_mrr.pt"))

    else:
        np.random.seed(random_seed)
        th.manual_seed(random_seed)

        graph = dgl_graph

        eids = np.arange(graph.num_edges)
        eids = np.random.permutation(eids)

        u, v = graph.edge_index

        test_size = int(len(eids) * test_ratio)
        val_size = int(len(eids) * val_ratio)

        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        val_pos_u, val_pos_v = (
            u[eids[test_size : test_size + val_size]],
            v[eids[test_size : test_size + val_size]],
        )
        train_pos_u, train_pos_v = (
            u[eids[test_size + val_size :]],
            v[eids[test_size + val_size :]],
        )

        valid_target_neg = th.randint(
            0, graph.num_nodes, [len(val_pos_u), int(neg_len)], dtype=th.long
        )  # TODO: filter some edges in graph
        test_target_neg = th.randint(
            0, graph.num_nodes, [len(test_pos_u), int(neg_len)], dtype=th.long
        )  # TODO: filter some edges in graph

        edge_split = {
            "train": {
                "source_node": train_pos_u,
                "target_node": train_pos_v,
                "edge": th.stack((train_pos_u, train_pos_v), dim=1),
                "train_edge_feature_index": eids[test_size + val_size :],
            },
            "valid": {
                "source_node": val_pos_u,
                "target_node": val_pos_v,
                "target_node_neg": valid_target_neg,
                "edge": th.stack((val_pos_u, val_pos_v), dim=1),
            },
            "test": {
                "source_node": test_pos_u,
                "target_node": test_pos_v,
                "target_node_neg": test_target_neg,
                "edge": th.stack((test_pos_u, test_pos_v), dim=1),
            },
        }

        th.save(edge_split, os.path.join(path, f"{neg_len}/edge_split_mrr.pt"))

    return edge_split


# TODO: Consider time in split
# def split_edge_MMR(dgl_graph, time=2015, random_seed=42, neg_len=1000, path=None):
#     if os.path.exists(os.path.join(path, 'edge_split_mrr.pt')):
#         edge_split = th.load(os.path.join(path, 'edge_split_mrr.pt'))
#     else:

#         np.random.seed(random_seed)
#         th.manual_seed(random_seed)

#         graph = from_dgl(dgl_graph)

#         year = list(np.array(graph.year))
#         indices = np.arange(graph.num_nodes)
#         all_source = graph.edge_index[0]
#         all_target = graph.edge_index[1]
#         import random
#         val_list = []
#         test_list = []
#         for i in range(year.index(time), graph.num_nodes):
#             indices = th.where(all_source == i)[0]
#             if len(indices) >= 2:
#                 _list = random.sample(list(indices), k=2)
#                 val_list.append(_list[0]) # TOOD: check edge
#                 test_list.append(_list[1])

#         val_source = all_source[val_list]
#         val_target = all_target[val_list]

#         test_source = all_source[test_list]
#         test_target = all_target[test_list]

#         all_index = list(range(0, len(graph.edge_index[0])))

#         val_list = np.array(val_list).tolist()
#         test_list = np.array(test_list).tolist()

#         train_idx = set(all_index) - set(val_list) - set(test_list)

#         tra_source = all_source[list(train_idx)]
#         tra_target = all_target[list(train_idx)]

#         val_target_neg = th.randint(low=0, high=year.index(time), size=(len(val_source), neg_len))
#         test_target_neg = th.randint(low=0, high=year.index(time), size=(len(test_source), neg_len))

#         # ! 创建dict类型存法
#         edge_split = {'train': {'source_node': tra_source, 'target_node': tra_target},
#                       'valid': {'source_node': val_source, 'target_node': val_target,
#                                 'target_node_neg': val_target_neg},
#                       'test': {'source_node': test_source, 'target_node': test_target,
#                                'target_node_neg': test_target_neg}}

#         th.save(edge_split, os.path.join(path, 'edge_split_mrr.pt'))
#         # ! 保存子图
#         train_g = dgl.remove_edges(dgl_graph, val_list + test_list)
#         dgl.save_graphs(os.path.join(path, 'train_G.pt'), train_g)

#     return edge_split, train_g


class Evaluator:
    def __init__(self, name):
        self.name = name
        meta_info = {
            "History": {"name": "History", "eval_metric": "hits@50"},
            "DBLP": {"name": "DBLP", "eval_metric": "mrr"},
        }

        self.eval_metric = meta_info[self.name]["eval_metric"]

        if "hits@" in self.eval_metric:
            ### Hits@K

            self.K = int(self.eval_metric.split("@")[1])

    def _parse_and_check_input(self, input_dict):
        if "hits@" in self.eval_metric:
            if not "y_pred_pos" in input_dict:
                raise RuntimeError("Missing key of y_pred_pos")
            if not "y_pred_neg" in input_dict:
                raise RuntimeError("Missing key of y_pred_neg")

            y_pred_pos, y_pred_neg = input_dict["y_pred_pos"], input_dict["y_pred_neg"]

            """
                y_pred_pos: numpy ndarray or torch tensor of shape (num_edge, )
                y_pred_neg: numpy ndarray or torch tensor of shape (num_edge, )
            """

            # convert y_pred_pos, y_pred_neg into either torch tensor or both numpy array
            # type_info stores information whether torch or numpy is used

            type_info = None

            # check the raw tyep of y_pred_pos
            if not (
                isinstance(y_pred_pos, np.ndarray)
                or (th is not None and isinstance(y_pred_pos, th.Tensor))
            ):
                raise ValueError(
                    "y_pred_pos needs to be either numpy ndarray or th tensor"
                )

            # check the raw type of y_pred_neg
            if not (
                isinstance(y_pred_neg, np.ndarray)
                or (th is not None and isinstance(y_pred_neg, th.Tensor))
            ):
                raise ValueError(
                    "y_pred_neg needs to be either numpy ndarray or th tensor"
                )

            # if either y_pred_pos or y_pred_neg is th tensor, use th tensor
            if th is not None and (
                isinstance(y_pred_pos, th.Tensor) or isinstance(y_pred_neg, th.Tensor)
            ):
                # converting to th.Tensor to numpy on cpu
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = th.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = th.from_numpy(y_pred_neg)

                # put both y_pred_pos and y_pred_neg on the same device
                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = "torch"

            else:
                # both y_pred_pos and y_pred_neg are numpy ndarray

                type_info = "numpy"

            if not y_pred_pos.ndim == 1:
                raise RuntimeError(
                    "y_pred_pos must to 1-dim arrray, {}-dim array given".format(
                        y_pred_pos.ndim
                    )
                )

            if not y_pred_neg.ndim == 1:
                raise RuntimeError(
                    "y_pred_neg must to 1-dim arrray, {}-dim array given".format(
                        y_pred_neg.ndim
                    )
                )

            return y_pred_pos, y_pred_neg, type_info

        elif "mrr" == self.eval_metric:

            if not "y_pred_pos" in input_dict:
                raise RuntimeError("Missing key of y_pred_pos")
            if not "y_pred_neg" in input_dict:
                raise RuntimeError("Missing key of y_pred_neg")

            y_pred_pos, y_pred_neg = input_dict["y_pred_pos"], input_dict["y_pred_neg"]

            """
                y_pred_pos: numpy ndarray or torch tensor of shape (num_edge, )
                y_pred_neg: numpy ndarray or torch tensor of shape (num_edge, num_node_negative)
            """

            # convert y_pred_pos, y_pred_neg into either torch tensor or both numpy array
            # type_info stores information whether torch or numpy is used

            type_info = None

            # check the raw tyep of y_pred_pos
            if not (
                isinstance(y_pred_pos, np.ndarray)
                or (th is not None and isinstance(y_pred_pos, th.Tensor))
            ):
                raise ValueError(
                    "y_pred_pos needs to be either numpy ndarray or th tensor"
                )

            # check the raw type of y_pred_neg
            if not (
                isinstance(y_pred_neg, np.ndarray)
                or (th is not None and isinstance(y_pred_neg, th.Tensor))
            ):
                raise ValueError(
                    "y_pred_neg needs to be either numpy ndarray or th tensor"
                )

            # if either y_pred_pos or y_pred_neg is th tensor, use th tensor
            if th is not None and (
                isinstance(y_pred_pos, th.Tensor) or isinstance(y_pred_neg, th.Tensor)
            ):
                # converting to th.Tensor to numpy on cpu
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = th.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = th.from_numpy(y_pred_neg)

                # put both y_pred_pos and y_pred_neg on the same device
                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = "torch"

            else:
                # both y_pred_pos and y_pred_neg are numpy ndarray

                type_info = "numpy"

            if not y_pred_pos.ndim == 1:
                raise RuntimeError(
                    "y_pred_pos must to 1-dim arrray, {}-dim array given".format(
                        y_pred_pos.ndim
                    )
                )

            if not y_pred_neg.ndim == 2:
                raise RuntimeError(
                    "y_pred_neg must to 2-dim arrray, {}-dim array given".format(
                        y_pred_neg.ndim
                    )
                )

            return y_pred_pos, y_pred_neg, type_info

        else:
            raise ValueError("Undefined eval metric %s" % (self.eval_metric))

    def eval(self, input_dict):

        if "hits@" in self.eval_metric:
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_hits(y_pred_pos, y_pred_neg, type_info)
        elif self.eval_metric == "mrr":
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_mrr(y_pred_pos, y_pred_neg, type_info)

        else:
            raise ValueError("Undefined eval metric %s" % (self.eval_metric))

    def _eval_hits(self, y_pred_pos, y_pred_neg, type_info):
        """
        compute Hits@K
        For each positive target node, the negative target nodes are the same.

        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
        """

        if len(y_pred_neg) < self.K:
            return {"hits@{}".format(self.K): 1.0}

        if type_info == "torch":
            kth_score_in_negative_edges = th.topk(y_pred_neg, self.K)[0][-1]
            hitsK = float(th.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(
                y_pred_pos
            )

        # type_info is numpy
        else:
            kth_score_in_negative_edges = np.sort(y_pred_neg)[-self.K]
            hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(
                y_pred_pos
            )

        return {"hits@{}".format(self.K): hitsK}

    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        """
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
        """

        if type_info == "torch":
            y_pred = th.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
            argsort = th.argsort(y_pred, dim=1, descending=True)
            ranking_list = th.nonzero(argsort == 0, as_tuple=False)
            ranking_list = ranking_list[:, 1] + 1
            # hits1_list = (ranking_list <= 1).to(th.float)
            # hits3_list = (ranking_list <= 3).to(th.float)
            # hits10_list = (ranking_list <= 10).to(th.float)
            mrr_list = 1.0 / ranking_list.to(th.float)

            return {
                # 'hits@1_list': hits1_list,
                #  'hits@3_list': hits3_list,
                #  'hits@10_list': hits10_list,
                "mrr_list": mrr_list
            }

        else:
            y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg], axis=1)
            argsort = np.argsort(-y_pred, axis=1)
            ranking_list = (argsort == 0).nonzero()
            ranking_list = ranking_list[1] + 1
            # hits1_list = (ranking_list <= 1).astype(np.float32)
            # hits3_list = (ranking_list <= 3).astype(np.float32)
            # hits10_list = (ranking_list <= 10).astype(np.float32)
            mrr_list = 1.0 / ranking_list.astype(np.float32)

            return {
                # 'hits@1_list': hits1_list,
                #  'hits@3_list': hits3_list,
                #  'hits@10_list': hits10_list,
                "mrr_list": mrr_list
            }
