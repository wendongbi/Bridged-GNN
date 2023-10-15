from builtins import NotImplementedError
from functools import reduce
import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch import Tensor
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul, set_diag
from torch_geometric.nn.conv import MessagePassing, gat_conv, gcn_conv, sage_conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import SplineConv, GATConv, GATv2Conv, SAGEConv, GCNConv, GCN2Conv, GENConv, DeepGCNLayer, APPNP, JumpingKnowledge, GINConv
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from typing import Union, Tuple, Optional
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch_sparse


class GINNet(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, hidden=16):
        super(GINNet, self).__init__()
        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(
                GINConv(Linear(dataset.num_features, dataset.num_classes))
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(GINConv(Linear(dataset.num_features, hidden), train_eps=True))
                elif num == layer_num - 1:
                    self.convs.append(GINConv(Linear(hidden, dataset.num_classes), train_eps=True))
                else:
                    self.convs.append(GINConv(Linear(hidden, hidden), train_eps=True))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, edge_index)
            else:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return F.log_softmax(x, dim=1)


class JKNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, mode='cat'):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        # self.bns = torch.nn.ModuleList()
        # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.jump = JumpingKnowledge(mode=mode, channels=hidden_channels, num_layers=num_layers)
        if mode == 'cat':
            self.lin = Linear(num_layers * hidden_channels, out_channels)
        else:
            self.lin = Linear(hidden_channels, out_channels)

        self.dropout = dropout
        self.adj_t_cache = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()

        self.jump.reset_parameters()
        self.lin.reset_parameters()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.adj_t_cache is None:
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
            self.adj_t_cache = gcn_norm(self.adj_t_cache)
            # self.adj_t_cache = adj_norm(self.adj_t_cache, norm='symmetric')
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, self.adj_t_cache)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x]

        x = self.jump(xs)
        x = self.lin(x)

        return F.log_softmax(x, dim=-1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, hidden=16):
        super().__init__()
        self.lin1 = Linear(dataset.num_features, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
        self.prop1 = APPNP(10, 0.1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

class DeeperGCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = Linear(dataset.num_features, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.node_encoder(x)

        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

class GCN2(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))
        self.adj_t_cache = None
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.adj_t_cache is None:
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
            self.adj_t_cache = gcn_norm(self.adj_t_cache)
            # self.adj_t_cache = adj_norm(self.adj_t_cache, norm='symmetric')
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, self.adj_t_cache)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return F.log_softmax(x, dim=1)

class ConvNet(torch.nn.Module):
    def __init__(self, dataset):
        super(ConvNet, self).__init__()
        self.conv1 = SplineConv(dataset.num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, dataset.num_classes, dim=1, kernel_size=2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(in_size, hidden)
        # self.hidden_layer = nn.Linear(hidden, hidden)
        self.out_layer = nn.Linear(hidden, out_size)
        self.reset_parameters()
        # self.adj = torch.sparse_coo_tensor([np.arange(0, 4040), np.arange(0, 4040)], torch.ones(4040), size=(4040, 4040)).requires_grad_(False)
        # self.adj = torch.eye(4040).to_sparse().requires_grad_(False)
    def reset_parameters(self):
        self.input_layer.reset_parameters()
        # self.hidden_layer.reset_parameters()
        self.out_layer.reset_parameters()
    def forward(self, data):
        x = data.x
        x = self.input_layer(x)
        x = F.dropout(F.relu(x), p=0.5, training=self.training)
        x = self.out_layer(x)
        logits = F.log_softmax(x, dim=1)
        return logits
    def get_emb(self, data):
        x = data.x
        x = self.input_layer(x)
        x = F.relu(x)
        return x
    def get_logits(self, data):
        x = data.x
        x = self.input_layer(x)
        x = F.relu(F.dropout(x, p=0.5, training=self.training))
        logits = self.out_layer(x)
        return logits

class GCNNet(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, hidden=16):
        super(GCNNet, self).__init__()
        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(
                GCNConv(dataset.num_features, dataset.num_classes)
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(GCNConv(dataset.num_features, hidden))
                elif num == layer_num - 1:
                    self.convs.append(GCNConv(hidden, dataset.num_classes))
                else:
                    self.convs.append(GCNConv(hidden, hidden))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, edge_index)
            else:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_emb(self, data):
        x, edge_index = data.x, data.edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                # x = conv(x, edge_index)
                continue
            else:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return x
    def get_logits(self, data):
        x, edge_index = data.x, data.edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, edge_index)
                # continue
            else:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return x

class GATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads,
                 dropout, att_dropout):
        super(GATv2, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=att_dropout, concat=True))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, dropout=att_dropout, concat=False))

        self.dropout = dropout
        self.adj_t_cache = None


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        assert isinstance(edge_index, torch.Tensor)
        if  self.adj_t_cache == None:
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
            # self.adj_t_cache = adj_norm(self.adj_t_cache, norm='row')

        # add self loop
        if isinstance(edge_index, Tensor):
            num_nodes = x.size(0)
            edge_index, edge_attr = remove_self_loops(
                edge_index)
            edge_index, edge_attr = add_self_loops(
                edge_index, fill_value='mean',
                num_nodes=num_nodes)
        elif isinstance(edge_index, SparseTensor):
            if self.edge_dim is None:
                edge_index = set_diag(edge_index)
            else:
                raise NotImplementedError(
                    "The usage of 'edge_attr' and 'add_self_loops' "
                    "simultaneously is currently not yet supported for "
                    "'edge_index' in a 'SparseTensor' form")

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index) # for GATv2Conv
            # x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index) # for GATv2Conv
        
        return x.log_softmax(dim=-1)


class MLP(torch.nn.Module):
    # MLP
    def __init__(self, dim_in, dim_out, dim_hidden=64, layer_num=2, root_weight=True, \
                 use_norm=False, norm_mode='PN-SCS', norm_scale=1, log_softmax=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        if layer_num == 1:
            self.layers.append(
                Linear(dim_in, dim_out, weight_initializer='glorot')
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.layers.append(Linear(dim_in, dim_hidden, weight_initializer='glorot'))
                elif num == layer_num - 1:
                    self.layers.append(Linear(dim_hidden, dim_out, weight_initializer='glorot'))
                else:
                    self.layers.append(Linear(dim_hidden, dim_hidden, weight_initializer='glorot'))
        if use_norm:
            self.norm = PairNorm(mode=norm_mode, scale=norm_scale)
        self.use_norm = use_norm
        self.log_softmax = log_softmax

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    

    def forward(self, x, edge_index):
        for ind, layer in enumerate(self.layers):
            if ind == len(self.layers) -1:
                x = layer(x)
            else:
                x = layer(x)
                if self.use_norm:
                    x = self.norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        # return feature without log_softmax/softmax
        if self.log_softmax:
            x = F.log_softmax(x, dim=1)
        return x

class GAT(torch.nn.Module):
    def __init__(self, dataset, hidden=16, head=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            hidden,
            heads=head,
            concat=True,
            dropout=0.6)
        self.conv2 = GATConv(
            hidden * head,
            dataset.num_classes,
            heads=1,
            concat=False,
            dropout=0.6)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_emb(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = self.conv2(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, hidden=16, root_weight=True):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(
                SAGEConv(dataset.num_features, dataset.num_classes, root_weight=root_weight)
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(SAGEConv(dataset.num_features, hidden, root_weight=root_weight))
                elif num == layer_num - 1:
                    self.convs.append(SAGEConv(hidden, dataset.num_classes, root_weight=root_weight))
                else:
                    self.convs.append(SAGEConv(hidden, hidden, root_weight=root_weight))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        adj_sp = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], value=torch.ones(edge_index.shape[1]).to(x.device), sparse_sizes=(x.shape[0], x.shape[0]))
        # adj_sp = edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, adj_sp)
            else:
                x = F.relu(conv(x, adj_sp))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_emb(self, data, layer_num=1):
        x, edge_index = data.x, data.edge_index
        adj_sp = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]).to(x.device), sparse_sizes=(x.shape[0], x.shape[0]))
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                # x = conv(x, adj_sp)
                continue
            else:
                x = F.relu(conv(x, adj_sp))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return x
    
    def get_logits(self, data, layer_num=1):
        x, edge_index = data.x, data.edge_index
        adj_sp = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]).to(x.device), sparse_sizes=(x.shape[0], x.shape[0]))
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, adj_sp)
            else:
                x = F.relu(conv(x, adj_sp))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return x


# def adj_norm(adj, norm='row'):
#     if not adj.has_value():
#         adj = adj.fill_value(1., dtype=None)
#     # add self loop
#     adj = fill_diag(adj, 1.)
#     deg = sparsesum(adj, dim=1)
#     if norm == 'symmetric':
#         deg_inv_sqrt = deg.pow_(-0.5)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
#         adj = mul(adj, deg_inv_sqrt.view(-1, 1)) # row normalization
#         adj = mul(adj, deg_inv_sqrt.view(1, -1)) # col normalization
#     elif norm == 'row':
#         deg_inv_sqrt = deg.pow_(-1)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
#         adj = mul(adj, deg_inv_sqrt.view(-1, 1)) # row normalization
#     else:
#         raise NotImplementedError('Not implete adj norm: {}'.format(norm))
#     return adj


def adj_norm(adj, norm='row'):
    if not adj.has_value():
        adj = adj.fill_value(1., dtype=None)
    # add self loop
    adj = fill_diag(adj, 0.)
    adj = fill_diag(adj, 1.)
    deg = sparsesum(adj, dim=1)
    if norm == 'symmetric':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj = mul(adj, deg_inv_sqrt.view(-1, 1)) # row normalization
        adj = mul(adj, deg_inv_sqrt.view(1, -1)) # col normalization
    elif norm == 'row':
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj = mul(adj, deg_inv_sqrt.view(-1, 1)) # row normalization
    else:
        raise NotImplementedError('Not implete adj norm: {}'.format(norm))
    return adj
