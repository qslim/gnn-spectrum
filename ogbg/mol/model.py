import torch
from torch import nn
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import function as fn
from dgl.ops.edge_softmax import edge_softmax
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogbg.mol.utils.mol_encoder import BondEncoder


class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_tasks,
                 num_basis):
        super(Net, self).__init__()

        self.layers = config.layers
        self.atom_encoder = AtomEncoder(emb_dim=config.hidden)
        self.bond_encoder = BondEncoder(emb_dim=config.hidden)

        self.convs = torch.nn.ModuleList()
        for i in range(config.layers):
            self.convs.append(Conv(hidden_size=config.hidden,
                                   dropout_rate=config.dropout))

        self.graph_pred_linear = torch.nn.Linear(config.hidden, num_tasks)

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
            nn.GELU(),
        )
        self.filter_drop = nn.Dropout(config.dropout)

    def forward(self, g, x, edge_attr, bases):
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        bases = self.filter_drop(self.filter_encoder(bases))
        bases = edge_softmax(g, bases)
        for conv in self.convs:
            x = conv(g, x, edge_attr, bases)
        h_graph = self.pool(g, x)
        return self.graph_pred_linear(h_graph)

    def __repr__(self):
        return self.__class__.__name__


class Conv(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Conv, self).__init__()
        self.pre_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )
        self.preffn_dropout = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, graph, x_feat, edge_attr, bases):
        with graph.local_scope():
            graph.ndata['x'] = x_feat
            graph.apply_edges(fn.copy_u('x', '_x'))
            xee = self.pre_ffn(graph.edata['_x'] + edge_attr) * bases
            graph.edata['v'] = xee
            graph.update_all(fn.copy_e('v', '_aggr_e'), fn.sum('_aggr_e', 'aggr_e'))
            y = graph.ndata['aggr_e']
            y = self.preffn_dropout(y)
            x = x_feat + y
            y = self.ffn(x)
            y = self.ffn_dropout(y)
            x = x + y
            return x
