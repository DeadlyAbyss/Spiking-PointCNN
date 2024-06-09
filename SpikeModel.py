import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Conv1d, Linear, Sequential, ELU
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

from typing import Optional

from torch_geometric.nn import fps, global_mean_pool
from torch_geometric.nn import Reshape
from torch_geometric.nn.inits import reset


try:
    from torch_cluster import knn_graph
except ImportError:
    knn_graph = None

import torch.nn.functional as F
import torch.nn as nn

from SpikingLayers import *

def add_T(tensor,Time_steps):
    tensor = tensor.unsqueeze(0).expand(Time_steps, -1, -1, -1)
    return tensor


class Spike_PointCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.Act = LIF_Act()

        self.conv1 = Spike_XConv(0, 48, dim=3, kernel_size=8, hidden_channels=32, lif_act=self.Act)
        self.conv2 = Spike_XConv(48, 96, dim=3, kernel_size=12, hidden_channels=64,
                               dilation=2, lif_act=self.Act)
        self.conv3 = Spike_XConv(96, 192, dim=3, kernel_size=16, hidden_channels=128,
                               dilation=2, lif_act=self.Act)
        self.conv4 = Spike_XConv(192, 384, dim=3, kernel_size=16,
                               hidden_channels=256, dilation=2, lif_act=self.Act)

        self.lin1 = SpikingLinear(Linear(384, 256))
        self.lin2 = SpikingLinear(Linear(256, 128))
        self.lin3 = SpikingLinear(Linear(128, num_classes))

    def forward(self, pos, batch):
        x = self.Act(self.conv1(None, pos, batch))

        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = self.Act(self.conv2(x, pos, batch))

        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = self.Act(self.conv3(x, pos, batch))
        x = self.Act(self.conv4(x, pos, batch))

        x = global_mean_pool(x, batch)

        x = self.Act(self.lin1(x))
        x = self.Act(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


class Spike_XConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dim: int,
                 kernel_size: int, hidden_channels: Optional[int] = None,
                 dilation: int = 1, bias: bool = True, num_workers: int = 1, lif_act=None):
        super().__init__()

        if knn_graph is None:
            raise ImportError('`XConv` requires `torch-cluster`.')

        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels // 4
        assert hidden_channels > 0
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_workers = num_workers
        self.Act = lif_act if lif_act is not None else LIF_Act()

        C_in, C_delta, C_out = in_channels, hidden_channels, out_channels
        D, K = dim, kernel_size

        self.mlp1 = Sequential(
            SpikingLinear(Linear(dim, C_delta)),
            self.Act,
            SpikingBatchNorm(BatchNorm1d(C_delta)),
            SpikingLinear(Linear(C_delta, C_delta)),
            self.Act,
            SpikingBatchNorm(BatchNorm1d(C_delta)),
            Reshape(-1, K, C_delta),
        )

        self.mlp2 = Sequential(
            SpikingLinear(Linear(D * K, K ** 2)),
            self.Act,
            SpikingBatchNorm(BatchNorm1d(K ** 2)),
            Reshape(-1, K, K),
            SpikingConv(Conv1d(K, K ** 2, K, groups=K)),
            self.Act,
            SpikingBatchNorm(BatchNorm1d(K ** 2)),
            Reshape(-1, K, K),
            SpikingConv(Conv1d(K, K ** 2, K, groups=K)),
            SpikingBatchNorm(BatchNorm1d(K ** 2)),
            Reshape(-1, K, K),
        )

        C_in = C_in + C_delta
        depth_multiplier = int(ceil(C_out / C_in))
        self.conv = Sequential(
            SpikingConv(Conv1d(C_in, C_in * depth_multiplier, K, groups=C_in)),
            Reshape(-1, C_in * depth_multiplier),
            SpikingLinear(Linear(C_in * depth_multiplier, C_out, bias=bias)),
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)
        reset(self.conv)

    def forward(self, x: Tensor, pos: Tensor, batch: Optional[Tensor] = None):
        r"""Runs the forward pass of the module."""
        pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
        (N, D), K = pos.size(), self.kernel_size

        edge_index = knn_graph(pos, K * self.dilation, batch, loop=True,
                               flow='target_to_source',
                               num_workers=self.num_workers)

        if self.dilation > 1:
            edge_index = edge_index[:, ::self.dilation]

        row, col = edge_index[0], edge_index[1]

        pos = pos[col] - pos[row]

        x_star = self.mlp1(pos)
        if x is not None:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            x = x[col].view(N, K, self.in_channels)
            x_star = torch.cat([x_star, x], dim=-1)
        x_star = x_star.transpose(1, 2).contiguous()

        transform_matrix = self.mlp2(pos.view(N, K * D))

        x_transformed = torch.matmul(x_star, transform_matrix)

        out = self.conv(x_transformed)

        return out
