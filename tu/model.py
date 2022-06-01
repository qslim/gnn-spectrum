import torch
from torch import nn
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import function as fn
from dgl.ops.edge_softmax import edge_softmax


class Net(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_basis,
                 config):
        super(Net, self).__init__()

        self.layers = config.layers
        self.lin0 = nn.Linear(input_dim, config.hidden)

        if config.nonlinear == 'ReLU':
            self.nonlinear = nn.ReLU()
        else:
            self.nonlinear = nn.GELU()

        self.convs = torch.nn.ModuleList()
        for i in range(config.layers):
            self.convs.append(Conv(hidden_size=config.hidden,
                                   dropout_rate=config.dropout,
                                   nonlinear=config.nonlinear))

        self.lin1 = nn.Linear(config.hidden, config.hidden)
        self.final_drop = nn.Dropout(config.dropout)
        self.lin2 = nn.Linear(config.hidden, output_dim)

        if config.pooling == 'S':
            self.pool = SumPooling()
        elif config.pooling == 'M':
            self.pool = AvgPooling()
        elif config.pooling == 'X':
            self.pool = MaxPooling()

        self.filter_encoder = nn.Sequential(
            nn.Linear(num_basis, config.hidden),
            nn.BatchNorm1d(config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.hidden),
            nn.BatchNorm1d(config.hidden),
            nn.GELU()
        )
        self.filter_drop = nn.Dropout(config.dropout)

    def forward(self, g, h, bases):
        h = self.lin0(h)
        bases = self.filter_drop(self.filter_encoder(bases))
        bases = edge_softmax(g, bases)
        for conv in self.convs:
            h = conv(g, h, bases)
        nr = self.pool(g, h)
        nr = self.nonlinear(self.lin1(nr))
        nr = self.final_drop(nr)
        nr = self.lin2(nr)
        return nr

    def __repr__(self):
        return self.__class__.__name__


class Conv(nn.Module):
    def __init__(self, hidden_size, dropout_rate, nonlinear):
        super(Conv, self).__init__()
        self.pre_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )
        self.preffn_dropout = nn.Dropout(dropout_rate)

        if nonlinear == 'ReLU':
            _nonlinear = nn.ReLU()
        else:
            _nonlinear = nn.GELU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            _nonlinear,
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            _nonlinear
        )
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, graph, x_feat, bases):
        with graph.local_scope():
            graph.ndata['x'] = self.pre_ffn(x_feat)
            graph.edata['v'] = bases
            graph.update_all(fn.u_mul_e('x', 'v', '_aggr_e'), fn.sum('_aggr_e', 'aggr_e'))
            y = graph.ndata['aggr_e']
            y = self.preffn_dropout(y)
            x = x_feat + y
            y = self.ffn(x)
            y = self.ffn_dropout(y)
            x = x + y
            return x
