import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn.dense.linear import Linear


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.mlp = ModuleList()
        self.mlp.append(Linear(in_channels, hidden_channels))
        if num_layers >= 2:
            for _ in range(num_layers - 2):
                self.mlp.append(Linear(hidden_channels, hidden_channels))
        self.mlp.append(Linear(hidden_channels, out_channels))

    def forward(self, x):
        for layer in self.mlp[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.mlp[-1](x)
        return x
