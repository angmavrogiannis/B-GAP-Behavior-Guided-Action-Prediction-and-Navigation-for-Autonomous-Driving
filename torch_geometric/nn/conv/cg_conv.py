from typing import Union, Tuple
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn.conv import MessagePassing


class CGConv(MessagePassing):
    r"""The crystal graph convolutional operator from the
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)}
        \sigma \left( \mathbf{z}_{i,j} \mathbf{W}_f + \mathbf{b}_f \right)
        \odot g \left( \mathbf{z}_{i,j} \mathbf{W}_s + \mathbf{b}_s  \right)

    where :math:`\mathbf{z}_{i,j} = [ \mathbf{x}_i, \mathbf{x}_j,
    \mathbf{e}_{i,j} ]` denotes the concatenation of central node features,
    neighboring node features and edge features.
    In addition, :math:`\sigma` and :math:`g` denote the sigmoid and softplus
    functions, respectively.

    Args:
        channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        dim (int, optional): Edge feature dimensionality. (default: :obj:`0`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        batch_norm (bool, optional): If set to :obj:`True`, will make use of
            batch normalization. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                 aggr: str = 'add', batch_norm: bool = False,
                 bias: bool = True, **kwargs):
        super(CGConv, self).__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.batch_norm = batch_norm

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = Linear(sum(channels) + dim, channels[1], bias=bias)
        self.lin_s = Linear(sum(channels) + dim, channels[1], bias=bias)
        self.bn = BatchNorm1d(channels[1])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.bn(out) if self.batch_norm else out
        out += x[1]
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)
