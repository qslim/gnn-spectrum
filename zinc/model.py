import torch
from torch import nn
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import function as fn


class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_tasks,
                 num_basis,
                 shared=False):
        super(Net, self).__init__()

        self.layers = config.layers
        self.atom_encoder = nn.Embedding(28, config.hidden)
        self.bond_encoder = nn.Embedding(4, config.hidden)

        self.convs = torch.nn.ModuleList()
        for i in range(config.layers):
            self.convs.append(Conv(hidden_size=config.hidden))

        self.graph_pred_linear = torch.nn.Linear(config.hidden, num_tasks)

        if config.pooling == 'S':
            self.pool = SumPooling()
        elif config.pooling == 'M':
            self.pool = AvgPooling()
        elif config.pooling == 'X':
            self.pool = MaxPooling()

        if shared:
            filter_n = 1
            print('Sharing the filter among signals.')
        else:
            filter_n = config.hidden
        self.filter_encoder = nn.Sequential(
            nn.Linear(num_basis, config.hidden),
            # nn.BatchNorm1d(config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, filter_n),
            # nn.BatchNorm1d(filter_n),
            nn.GELU(),
        )

    def forward(self, g, x, edge_attr, bases):
        x = self.atom_encoder(x)
        # edge_attr = self.bond_encoder(edge_attr).sum(dim=1)
        edge_attr = self.bond_encoder(edge_attr)
        # edge_attr = self.bond_encoder(edge_attr[:, 0])
        bases = self.filter_encoder(bases)
        for conv in self.convs:
            x = conv(g, x, edge_attr, bases)
        h_graph = self.pool(g, x)
        return self.graph_pred_linear(h_graph)

    def __repr__(self):
        return self.__class__.__name__


class Conv(nn.Module):
    def __init__(self, hidden_size):
        super(Conv, self).__init__()
        self.pre_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )

    def forward(self, graph, x_feat, edge_attr, bases):
        with graph.local_scope():
            graph.ndata['x'] = x_feat
            graph.edata['e'] = edge_attr
            graph.apply_edges(fn.u_add_e('x', 'e', 'pos_e'))
            graph.edata['v'] = self.pre_ffn(graph.edata['pos_e']) * bases
            graph.update_all(fn.copy_e('v', 'pre_aggr'), fn.sum('pre_aggr', 'aggr'))
            y = graph.ndata['aggr']
            x = x_feat + y
            y = self.ffn(x)
            x = x + y
            return x
