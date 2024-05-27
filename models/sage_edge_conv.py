"""
sage_edge_conv.py includes edge_attr to graphsage
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul


class SAGEEdgeConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            edge_dim: int,
            aggr: str = 'mean',
            normalize: bool = False,
            root_weight: bool = True,
            project: bool = False,
            bias: bool = True,
            **kwargs,
    ):
        kwargs['aggr'] = aggr if aggr != 'lstm' else None
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        self.lin_t = Linear(edge_dim, in_channels[0], bias=bias)
        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        if self.aggr is None:
            self.lstm.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return x_j + self.lin_t(edge_attr)

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def aggregate(self, x: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        if self.aggr is not None:
            return scatter(x, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

        # LSTM aggregation:
        if ptr is None and not torch.all(index[:-1] <= index[1:]):
            raise ValueError(f"Can not utilize LSTM-style aggregation inside "
                             f"'{self.__class__.__name__}' in case the "
                             f"'edge_index' tensor is not sorted by columns. "
                             f"Run 'sort_edge_index(..., sort_by_row=False)' "
                             f"in a pre-processing step.")

        x, mask = to_dense_batch(x, batch=index, batch_size=dim_size)
        out, _ = self.lstm(x)
        return out[:, -1]

    def __repr__(self) -> str:
        aggr = self.aggr if self.aggr is not None else 'lstm'
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={aggr})')
