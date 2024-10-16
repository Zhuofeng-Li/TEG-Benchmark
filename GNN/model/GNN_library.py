import torch
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    TransformerConv,
    GATConv,
    GINEConv,
    GeneralConv, 
)
from torch.nn import Linear
import torch.nn.functional as F
import torch
import torch.nn as nn

import dgl.nn.pytorch as dglnn
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import torch.nn.functional as F
from dgl.sampling import node2vec_random_walk
from torch.utils.data import DataLoader
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class APPNP(nn.Module):
    def __init__(
        self,
        in_feats,
        n_layers,
        n_hidden,
        n_classes,
        activation,
        input_drop,
        edge_drop,
        alpha,
        k,
    ):
        super(APPNP, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        # input layer
        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            self.layers.append(nn.Linear(in_hidden, out_hidden))

        self.activation = activation

        self.input_drop = nn.Dropout(input_drop)

        self.propagate = dglnn.APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph, features):
        # prediction step
        h = features
        h = self.input_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.input_drop(h))
        # propagation step
        h = self.propagate(graph, h)
        return h


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        activation,
        dropout,
        aggregator_type,
        input_drop=0.0,
    ):
        super(GraphSAGE, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # input layer
        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes

            self.layers.append(dglnn.SAGEConv(in_hidden, out_hidden, aggregator_type))
            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for l, layer in enumerate(self.layers):
            conv = layer(graph, h)
            h = conv

            if l != len(self.layers) - 1:
                h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GAT(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        n_heads,
        activation,
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        use_attn_dst=True,
        use_symmetric_norm=False,
        residual=False,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            num_heads = n_heads if i < n_layers - 1 else 1
            out_channels = n_heads

            self.convs.append(
                GATConv(
                    in_hidden,
                    out_hidden,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    use_symmetric_norm=use_symmetric_norm,
                    residual=residual,
                )
            )

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = ElementWiseLinear(
            n_classes, weight=False, bias=True, inplace=True
        )

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            h = conv

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        h = h.mean(1)
        h = self.bias_last(h)

        return h


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(
        self, num_layers, input_dim, hidden_dim, output_dim, input_drop=0.0, dropout=0.2
    ):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            x = self.input_drop(x)
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
                h = self.dropout(h)
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""

    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        num_mlp_layers,
        input_dropout,
        learn_eps,
        neighbor_pooling_type,
    ):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(GIN, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.n_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, in_feats, n_hidden, n_hidden)
            else:
                mlp = MLP(num_mlp_layers, n_hidden, n_hidden, n_hidden)

            self.ginlayers.append(
                dglnn.GINConv(
                    ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(n_hidden))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(n_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(in_feats, n_classes))
            else:
                self.linears_prediction.append(nn.Linear(n_hidden, n_classes))

        self.input_drop = nn.Dropout(input_dropout)

    def forward(self, g, h):
        h = self.input_drop(h)
        # list of hidden representation at each layer (including input)
        hidden_rep = []

        for i in range(self.n_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        z = torch.cat(hidden_rep, dim=1)

        return z


class JKNet(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers=1, mode="cat", dropout=0.0
    ):
        super(JKNet, self).__init__()

        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, activation=F.relu))
        for _ in range(n_layers):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, activation=F.relu))

        if self.mode == "lstm":
            self.jump = dglnn.JumpingKnowledge(mode, n_hidden, n_layers)
        else:
            self.jump = dglnn.JumpingKnowledge(mode)

        if self.mode == "cat":
            n_hidden = n_hidden * (n_layers + 1)

        self.output = nn.Linear(n_hidden, n_classes)
        self.reset_params()

    def reset_params(self):
        self.output.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()
        if self.mode == "lstm":
            self.lstm.reset_parameters()
            self.attn.reset_parameters()

    def forward(self, g, feats):
        feat_lst = []
        for layer in self.layers:
            feats = self.dropout(layer(g, feats))
            feat_lst.append(feats)

        if self.mode == "cat":
            out = torch.cat(feat_lst, dim=-1)
        elif self.mode == "max":
            out = torch.stack(feat_lst, dim=-1).max(dim=-1)[0]
        else:
            # lstm
            x = torch.stack(feat_lst, dim=1)
            alpha, _ = self.lstm(x)
            alpha = self.attn(alpha).squeeze(-1)
            alpha = torch.softmax(alpha, dim=-1).unsqueeze(-1)
            out = (x * alpha).sum(dim=1)

        g.ndata["h"] = out
        g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))

        return self.output(g.ndata["h"])


class MoNet(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        dim,
        n_kernels,
        input_drop,
        dropout,
    ):
        super(MoNet, self).__init__()
        self.input_drop = nn.Dropout(input_drop)
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Input layer
        self.layers.append(dglnn.GMMConv(in_feats, n_hidden, dim, n_kernels))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Hidden layer
        for _ in range(n_layers - 1):
            self.layers.append(dglnn.GMMConv(n_hidden, n_hidden, dim, n_kernels))
            self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Output layer
        self.layers.append(dglnn.GMMConv(n_hidden, n_classes, dim, n_kernels))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat, pseudo, graph):
        h = feat
        h = self.input_drop(h)
        for i in range(len(self.layers)):
            if i != 0:
                h = self.dropout(h)
            h = self.layers[i](graph, h, self.pseudo_proj[i](pseudo))
        return h


class Node2vec(nn.Module):
    """Node2vec model from paper node2vec: Scalable Feature Learning for Networks <https://arxiv.org/abs/1607.00653>
    Attributes
    ----------
    g: DGLGraph
        The graph.
    embedding_dim: int
        Dimension of node embedding.
    walk_length: int
        Length of each trace.
    p: float
        Likelihood of immediately revisiting a node in the walk.  Same notation as in the paper.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
        Same notation as in the paper.
    num_walks: int
        Number of random walks for each node. Default: 10.
    window_size: int
        Maximum distance between the center node and predicted node. Default: 5.
    num_negatives: int
        The number of negative samples for each positive sample.  Default: 5.
    use_sparse: bool
        If set to True, use PyTorch's sparse embedding and optimizer. Default: ``True``.
    weight_name : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.
        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.
        If omitted, DGL assumes that the neighbors are picked uniformly.
    """

    def __init__(
        self,
        g,
        embedding_dim,
        walk_length,
        p,
        q,
        num_walks=10,
        window_size=5,
        num_negatives=5,
        use_sparse=True,
        weight_name=None,
    ):
        super(Node2vec, self).__init__()

        assert walk_length >= window_size

        self.g = g
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.N = self.g.num_nodes()
        if weight_name is not None:
            self.prob = weight_name
        else:
            self.prob = None

        self.embedding = nn.Embedding(self.N, embedding_dim, sparse=use_sparse)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def sample(self, batch):
        """
        Generate positive and negative samples.
        Positive samples are generated from random walk
        Negative samples are generated from random sampling
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch = batch.repeat(self.num_walks)
        # positive
        pos_traces = node2vec_random_walk(
            self.g, batch, self.p, self.q, self.walk_length, self.prob
        )
        pos_traces = pos_traces.unfold(1, self.window_size, 1)  # rolling window
        pos_traces = pos_traces.contiguous().view(-1, self.window_size)

        # negative
        neg_batch = batch.repeat(self.num_negatives)
        neg_traces = torch.randint(self.N, (neg_batch.size(0), self.walk_length))
        neg_traces = torch.cat([neg_batch.view(-1, 1), neg_traces], dim=-1)
        neg_traces = neg_traces.unfold(1, self.window_size, 1)  # rolling window
        neg_traces = neg_traces.contiguous().view(-1, self.window_size)

        return pos_traces, neg_traces

    def forward(self, nodes=None):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.
        Returns
        -------
        Tensor
            Node embedding
        """
        emb = self.embedding.weight
        if nodes is None:
            return emb
        else:
            return emb[nodes]

    def loss(self, pos_trace, neg_trace):
        """
        Computes the loss given positive and negative random walks.
        Parameters
        ----------
        pos_trace: Tensor
            positive random walk trace
        neg_trace: Tensor
            negative random walk trace
        """
        e = 1e-15

        # Positive
        pos_start, pos_rest = (
            pos_trace[:, 0],
            pos_trace[:, 1:].contiguous(),
        )  # start node and following trace
        w_start = self.embedding(pos_start).unsqueeze(dim=1)
        w_rest = self.embedding(pos_rest)
        pos_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # Negative
        neg_start, neg_rest = neg_trace[:, 0], neg_trace[:, 1:].contiguous()

        w_start = self.embedding(neg_start).unsqueeze(dim=1)
        w_rest = self.embedding(neg_rest)
        neg_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # compute loss
        pos_loss = -torch.log(torch.sigmoid(pos_out) + e).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + e).mean()

        return pos_loss + neg_loss

    def loader(self, batch_size):
        """
        Parameters
        ----------
        batch_size: int
            batch size
        Returns
        -------
        DataLoader
            Node2vec training data loader
        """
        return DataLoader(
            torch.arange(self.N),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.sample,
        )


class Node2vecModel(object):
    """
    Wrapper of the ``Node2Vec`` class with a ``train`` method.
    Attributes
    ----------
    g: DGLGraph
        The graph.
    embedding_dim: int
        Dimension of node embedding.
    walk_length: int
        Length of each trace.
    p: float
        Likelihood of immediately revisiting a node in the walk.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
    num_walks: int
        Number of random walks for each node. Default: 10.
    window_size: int
        Maximum distance between the center node and predicted node. Default: 5.
    num_negatives: int
        The number of negative samples for each positive sample.  Default: 5.
    use_sparse: bool
        If set to True, uses PyTorch's sparse embedding and optimizer. Default: ``True``.
    weight_name : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.
        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.
        If omitted, DGL assumes that the neighbors are picked uniformly. Default: ``None``.
    eval_set: list of tuples (Tensor, Tensor)
        [(nodes_train,y_train),(nodes_val,y_val)]
        If omitted, model will not be evaluated. Default: ``None``.
    eval_steps: int
        Interval steps of evaluation.
        if set <= 0, model will not be evaluated. Default: ``None``.
    device: str
        device, default 'cpu'.
    """

    def __init__(
        self,
        g,
        embedding_dim,
        walk_length,
        p=1.0,
        q=1.0,
        num_walks=1,
        window_size=5,
        num_negatives=5,
        use_sparse=True,
        weight_name=None,
        eval_set=None,
        eval_steps=-1,
        device="cpu",
    ):

        self.model = Node2vec(
            g,
            embedding_dim,
            walk_length,
            p,
            q,
            num_walks,
            window_size,
            num_negatives,
            use_sparse,
            weight_name,
        )
        self.g = g
        self.use_sparse = use_sparse
        self.eval_steps = eval_steps
        self.eval_set = eval_set

        if device == "cpu":
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _train_step(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        for pos_traces, neg_traces in loader:
            pos_traces, neg_traces = pos_traces.to(device), neg_traces.to(device)
            optimizer.zero_grad()
            loss = model.loss(pos_traces, neg_traces)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def _evaluate_step(self):
        nodes_train, y_train = self.eval_set[0]
        nodes_val, y_val = self.eval_set[1]

        acc = self.model.evaluate(nodes_train, y_train, nodes_val, y_val)
        return acc

    def train(self, epochs, batch_size, learning_rate=0.01):
        """
        Parameters
        ----------
        epochs: int
            num of train epoch
        batch_size: int
            batch size
        learning_rate: float
            learning rate. Default 0.01.
        """

        self.model = self.model.to(self.device)
        loader = self.model.loader(batch_size)
        if self.use_sparse:
            optimizer = torch.optim.SparseAdam(
                list(self.model.parameters()), lr=learning_rate
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for i in range(epochs):
            loss = self._train_step(self.model, loader, optimizer, self.device)
            if self.eval_steps > 0:
                if epochs % self.eval_steps == 0:
                    acc = self._evaluate_step()
                    print(
                        "Epoch: {}, Train Loss: {:.4f}, Val Acc: {:.4f}".format(
                            i, loss, acc
                        )
                    )

    def embedding(self, nodes=None):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.
        Returns
        -------
        Tensor
            Node embedding.
        """

        return self.model(nodes)


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        edge_dim,
        hidden_channels,
        out_channels,
        num_layers,
        heads,
        dropout,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads,
                edge_dim=edge_dim,
                add_self_loops=False,
            )
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    heads * hidden_channels,
                    hidden_channels,
                    heads,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
            )
        self.convs.append(
            GATConv(
                heads * hidden_channels,
                out_channels,
                heads,
                concat=False,
                edge_dim=edge_dim,
                add_self_loops=False,
            )
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphTransformer(torch.nn.Module):
    def __init__(
        self, in_channels, edge_dim, hidden_channels, out_channels, num_layers, dropout
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
            )
        self.convs.append(
            TransformerConv(hidden_channels, out_channels, edge_dim=edge_dim)
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GINE(torch.nn.Module):
    def __init__(
        self, in_channels, edge_dim, hidden_channels, out_channels, num_layers, dropout
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GINEConv(Linear(in_channels, hidden_channels), edge_dim=edge_dim)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GINEConv(Linear(hidden_channels, hidden_channels), edge_dim=edge_dim)
            )
        self.convs.append(
            GINEConv(Linear(hidden_channels, out_channels), edge_dim=edge_dim)
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class EdgeConvConv(MessagePassing):
    def __init__(
        self,
        nn: Callable,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn, "in_features"):
                in_channels = self.nn.in_features
            else:
                in_channels = self.nn.in_channels
            self.lin = Linear(edge_dim, in_channels)
        else:
            self.lin = None

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:

        temp = torch.cat([x_i, x_j, edge_attr], dim=1)

        return self.nn(temp)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


class EdgeConv(torch.nn.Module):
    def __init__(
        self, in_channels, edge_dim, hidden_channels, out_channels, num_layers, dropout
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            EdgeConvConv(
                Linear(2 * in_channels + edge_dim, hidden_channels), edge_dim=edge_dim
            )
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                EdgeConvConv(
                    Linear(2 * hidden_channels + edge_dim, hidden_channels),
                    edge_dim=edge_dim,
                )
            )
        self.convs.append(
            EdgeConvConv(
                Linear(2 * hidden_channels + edge_dim, out_channels), edge_dim=edge_dim
            )
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GeneralGNN(torch.nn.Module):
    def __init__(
        self, in_channels, edge_dim, hidden_channels, out_channels, num_layers, dropout
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GeneralConv(in_channels, hidden_channels, in_edge_channels=edge_dim)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GeneralConv(hidden_channels, hidden_channels, in_edge_channels=edge_dim)
            )
        self.convs.append(
            GeneralConv(hidden_channels, out_channels, in_edge_channels=edge_dim)
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
