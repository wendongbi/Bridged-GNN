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
import random




class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=10):
        """
            mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
            ('SCS'-mode is not in the paper but we found it works well in practice,
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x
    

class Similar(torch.nn.Module):
    def __init__(self, in_channels, num_clf_classes, dropout=0.6, use_clf=True):
        super(Similar, self).__init__()
        self.biasatt = nn.Sequential(
            Linear(128, 64, bias=True, weight_initializer='glorot'),
            nn.Tanh(),   
            Linear(64, 128, bias=True, weight_initializer='glorot'),
        )

        for m in self.biasatt:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # self.lin_input = nn.Sequential(
        #     Linear(in_channels, 512, bias=True, weight_initializer='glorot'),
        #     nn.Tanh(),
        #     Linear(512, 256, bias=True, weight_initializer='glorot'),
        #     nn.Tanh(),
        #     Linear(256, 128, bias=True, weight_initializer='glorot'),
        # )

        self.use_clf = use_clf
        if use_clf:
            self.lin_clf = Linear(in_channels, num_clf_classes, bias=True, weight_initializer='glorot')

        self.lin_self = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            Linear(in_channels, 64, bias=False, weight_initializer='glorot'),
            nn.BatchNorm1d(64),
            nn.Tanh(),  
            Linear(64, 128, bias=False, weight_initializer='glorot'),
        )

        # self.lin_neb = nn.Sequential(
        #     Linear(128, 64, bias=False, weight_initializer='glorot'),
        #     nn.Tanh(),  
        #     Linear(64, 128, bias=False, weight_initializer='glorot'),
        # )
        
        self.dropout = dropout
        # self.adj_t_cache = None
        self.reset_parameters()
    def reset_parameters(self):
        if self.use_clf:
            self.lin_clf.reset_parameters()
        for m in self.lin_self:
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight)
                m.reset_parameters()
        # for m in self.lin_neb:
        #     if isinstance(m, nn.Linear):
        #         # nn.init.kaiming_normal_(m.weight)
        #         m.reset_parameters()
        # for m in self.lin_input:
        #     if isinstance(m, nn.Linear):
        #         m.reset_parameters()
    def similarity_cross_domain(self, x_src, x_tar, idx1, idx2) -> Tensor:
        z_src = self.lin_self(x_src)
        z_tar = self.lin_self(x_tar)
        alpha = torch.nn.CosineSimilarity(dim=1)(z_src[idx1]+ self.biasatt(z_src[idx1]), z_tar[idx2]+ self.biasatt(z_tar[idx2]))
        # alpha = torch.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training)*5)
        alpha = torch.sigmoid(alpha)
        return alpha

    def forward_cross_domain(self, x_src, x_tar, idx1, idx2):
        z_src, z_tar = x_src, x_tar
        #classifier
        log_probs_clf_src = log_probs_clf_tar = None
        if self.use_clf:
            logits_src = self.lin_clf(F.dropout(F.relu(z_src), p=self.dropout, training=self.training))
            logits_tar = self.lin_clf(F.dropout(F.relu(z_tar), p=self.dropout, training=self.training))
            log_probs_clf_src = F.log_softmax(logits_src, dim=-1)
            log_probs_clf_tar = F.log_softmax(logits_tar, dim=-1)
        alpha = self.similarity_cross_domain(z_src, z_tar, idx1, idx2)
        return alpha.unsqueeze(-1), log_probs_clf_src, log_probs_clf_tar

    def similarity(self, x, idx1, idx2) -> Tensor:
        z = self.lin_self(x)
        alpha = torch.nn.CosineSimilarity(dim=1)(z[idx1]+ self.biasatt(z[idx1]), z[idx2]+ self.biasatt(z[idx2]))
        # alpha = torch.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training)*5)
        alpha = torch.sigmoid(alpha)
        return alpha

    def forward(self, x, idx1, idx2):
        # x, edge_index = data.x, data.edge_index
        
        # assert isinstance(edge_index, torch.Tensor)
        # if  self.adj_t_cache == None:
        #     self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))

        # adj_t=self.adj_t_cache
        # z = self.lin_input(x)
        z = x
        # mu_neb = matmul(adj_t, z, reduce='mean')

        #classifier
        log_probs_clf = None
        if self.use_clf:
            logits = self.lin_clf(F.dropout(F.relu(z), p=self.dropout, training=self.training))
            log_probs_clf = F.log_softmax(logits, dim=-1)
        alpha = self.similarity(z, idx1, idx2)
        return alpha.unsqueeze(-1), log_probs_clf
    
class Similar_noTrans(torch.nn.Module):
    def __init__(self, in_channels, num_clf_classes, dropout=0.6, use_clf=True):
        super(Similar_noTrans, self).__init__()
        self.use_clf = use_clf
        if use_clf:
            self.lin_clf = Linear(in_channels, num_clf_classes, bias=True, weight_initializer='glorot')

        
        self.dropout = dropout
        # self.adj_t_cache = None
        self.reset_parameters()
    def reset_parameters(self):
        if self.use_clf:
            self.lin_clf.reset_parameters()
    def similarity_cross_domain(self, x_src, x_tar, idx1, idx2) -> Tensor:
        alpha = torch.nn.CosineSimilarity(dim=1)(x_src[idx1], x_tar[idx2])
        # alpha = torch.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training)*5)
        alpha = torch.sigmoid(alpha)
        return alpha

    def forward_cross_domain(self, x_src, x_tar, idx1, idx2):
        z_src, z_tar = x_src, x_tar
        #classifier
        log_probs_clf_src = log_probs_clf_tar = None
        if self.use_clf:
            logits_src = self.lin_clf(F.dropout(F.relu(z_src), p=self.dropout, training=self.training))
            logits_tar = self.lin_clf(F.dropout(F.relu(z_tar), p=self.dropout, training=self.training))
            log_probs_clf_src = F.log_softmax(logits_src, dim=-1)
            log_probs_clf_tar = F.log_softmax(logits_tar, dim=-1)
        alpha = self.similarity_cross_domain(z_src, z_tar, idx1, idx2)
        return alpha.unsqueeze(-1), log_probs_clf_src, log_probs_clf_tar

    def similarity(self, x, idx1, idx2) -> Tensor:
        alpha = torch.nn.CosineSimilarity(dim=1)(x[idx1], x[idx2])
        # alpha = torch.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training)*5)
        alpha = torch.sigmoid(alpha)
        return alpha

    def forward(self, x, idx1, idx2):
        z = x
        #classifier
        log_probs_clf = None
        if self.use_clf:
            logits = self.lin_clf(F.dropout(F.relu(z), p=self.dropout, training=self.training))
            log_probs_clf = F.log_softmax(logits, dim=-1)
        alpha = self.similarity(z, idx1, idx2)
        return alpha.unsqueeze(-1), log_probs_clf
    
    
class GraphEncoder(torch.nn.Module):
    # GraphSAGE
    def __init__(self, dim_in, dim_out, dim_hidden=64, layer_num=2, root_weight=True, norm_mode='PN-SCS', norm_scale=1, log_softmax=False):
        super(GraphEncoder, self).__init__()
        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(
                SAGEConv(dim_in, dim_out, root_weight=root_weight)
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(SAGEConv(dim_in, dim_hidden, root_weight=root_weight))
                elif num == layer_num - 1:
                    self.convs.append(SAGEConv(dim_hidden, dim_out, root_weight=root_weight))
                else:
                    self.convs.append(SAGEConv(dim_hidden, dim_hidden, root_weight=root_weight))
        self.norm = PairNorm(mode=norm_mode, scale=norm_scale)
        self.log_softmax = log_softmax

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    

    def forward(self, x, edge_index):
        # adj_sp = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], value=torch.ones(edge_index.shape[1]).to(x.device), sparse_sizes=(x.shape[0], x.shape[0]))
        adj_sp = edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, adj_sp)
                # print(conv.lin_l.weight.grad)
            else:
                x = conv(x, adj_sp)
                # print('1', x.mean())
                x = self.norm(x)
                # print('2', x.mean())
                x = F.relu(x)
                # x  = F.leaky_relu(x, negative_slope=0.2)
                x = F.dropout(x, p=0.5, training=self.training)
        # return feature without log_softmax/softmax
        if self.log_softmax:
            x = F.log_softmax(x, dim=1)
        return x

def pair_enumeration(x1, x2):
    '''
        input:  [B,D]
        return: [B*B,D]
        input  [[a],
                [b]]
        return [[a,a],
                [b,a],
                [a,b],
                [b,b]]
    '''
    assert x1.ndimension() == 2 and x2.ndimension() == 2, 'Input dimension must be 2'
    # [a,b,c,a,b,c,a,b,c]
    # [a,a,a,b,b,b,c,c,c]
    x1_ = x1.repeat(x2.size(0), 1)
    x2_ = x2.repeat(1, x1.size(0)).view(-1, x1.size(1))
    # print(x1_, x2_)
    return torch.cat((x1_, x2_), dim=1)

class Pair_Enumerator_cross:
    # random&balance generate pair indexes
    def __init__(self, data_src, data_tar, mode='train'):
        super(Pair_Enumerator_cross, self).__init__()
        self.num_classes = data_src.y.max().item() + 1 # y start from 0, -1 denotes missing
        self.class_bucket_src = {} # class->indexes
        self.class_bucket_tar = {} # class->indexes
        idx2lbl_src = torch.Tensor(list(enumerate(data_src.y.tolist()))).long().transpose(0, 1) # tmp variable
        idx2lbl_tar = torch.Tensor(list(enumerate(data_tar.y.tolist()))).long().transpose(0, 1) # tmp variable
        self.mode = mode
        for lbl in range(self.num_classes):
            if mode == 'train':
                self.class_bucket_src[lbl] = idx2lbl_src[0][(idx2lbl_src[1] == lbl) * data_src.train_mask.cpu()]
                self.class_bucket_tar[lbl] = idx2lbl_tar[0][(idx2lbl_tar[1] == lbl) * data_tar.train_mask.cpu()]
            elif mode == 'val':
                self.class_bucket_src[lbl] = idx2lbl_src[0][(idx2lbl_src[1] == lbl) * data_src.val_mask.cpu()]
                self.class_bucket_tar[lbl] = idx2lbl_tar[0][(idx2lbl_tar[1] == lbl) * data_tar.val_mask.cpu()]
            elif mode == 'test':
                self.class_bucket_src[lbl] = idx2lbl_src[0][(idx2lbl_src[1] == lbl) * data_src.test_mask.cpu()]
                self.class_bucket_tar[lbl] = idx2lbl_tar[0][(idx2lbl_tar[1] == lbl) * data_tar.test_mask.cpu()]
            elif mode == 'all':
                mask_all_src = (data_src.train_mask + data_src.val_mask + data_src.test_mask).cpu()
                self.class_bucket_src[lbl] = idx2lbl_src[0][idx2lbl_src[1] == lbl * mask_all_src]
                mask_all_tar = (data_tar.train_mask + data_tar.val_mask + data_tar.test_mask).cpu()
                self.class_bucket_tar[lbl] = idx2lbl_tar[0][idx2lbl_tar[1] == lbl * mask_all_tar]
            else:
                raise NotImplementedError('Not Implemented Mode:{}'.format(mode))
    def balanced_sampling(self, max_class_num=2, sample_size=10000, shuffle=True):
        # x may be raw_feature/hidden_feature
        # sample max_class_num among all classes from which to conduct sampling, to keep label-balance
        
        if self.num_classes > max_class_num:
            selected_classes = np.random.choice(torch.arange(self.num_classes), replace=False, size=max_class_num) # sampling classes without putback
        else:
            selected_classes = np.arange(self.num_classes).astype(np.int8)
        # same-class pair generation
        sample_idxs_1 = []
        sample_idxs_2 = []
        sample_per_class_same = int(0.5 * sample_size / max_class_num)
        sample_per_class_diff = int(0.5 * sample_size / (max_class_num * (max_class_num - 1)))
        for lbl_1 in selected_classes:
            for lbl_2 in selected_classes:
                if lbl_1 == lbl_2:
                    idx1 = torch.from_numpy(np.random.choice(self.class_bucket_src[lbl_1], size=sample_per_class_same)).long() # sampling with putback
                    idx2 = torch.from_numpy(np.random.choice(self.class_bucket_tar[lbl_2], size=sample_per_class_same)).long() # sampling with putback
                else:
                    idx1 = torch.from_numpy(np.random.choice(self.class_bucket_src[lbl_1], size=sample_per_class_diff)).long() # sampling with putback
                    idx2 = torch.from_numpy(np.random.choice(self.class_bucket_tar[lbl_2], size=sample_per_class_diff)).long() # sampling with putback
                # pair_idxs = torch.stack((idx1, idx2), dim=0) # 2 * pair_num
                sample_idxs_1.append(idx1) 
                sample_idxs_2.append(idx2)
        sample_idxs_1 = torch.cat(sample_idxs_1, dim=0)
        sample_idxs_2 = torch.cat(sample_idxs_2, dim=0)

        # shuffle
        if shuffle:
            shuffle_idx = torch.arange(sample_idxs_1.shape[0]).tolist()
            random.shuffle(shuffle_idx)
            sample_idxs_1 = sample_idxs_1[shuffle_idx]
            sample_idxs_2 = sample_idxs_1[shuffle_idx]
        # # x_pair = torch.cat([x[sample_idxs_1], x[sample_idxs_2]], dim=1) # need to calculate gradient for this step
        # # y_pair = (y[sample_idxs_1] == y[sample_idxs_2]).long()
        return sample_idxs_1, sample_idxs_2
    def sampling(self, max_class_num=2, sample_size=10000, shuffle=True):
        # x may be raw_feature/hidden_feature
        # sample max_class_num among all classes from which to conduct sampling, to keep label-balance
        
        if self.num_classes > max_class_num:
            selected_classes = np.random.choice(torch.arange(self.num_classes), replace=False, size=max_class_num) # sampling classes without putback
        else:
            selected_classes = np.arange(self.num_classes).astype(np.int8)
        sample_idxs_1 = []
        sample_idxs_2 = []
        sample_per_class = int(np.sqrt(sample_size) / max_class_num)
        for lbl in selected_classes:
            sample_idxs_1.append(torch.from_numpy(np.random.choice(self.class_bucket_src[lbl], size=sample_per_class)).long()) # sampling with putback
            sample_idxs_2.append(torch.from_numpy(np.random.choice(self.class_bucket_tar[lbl], size=sample_per_class)).long()) # sampling with putback
        sample_idxs_1 = torch.cat(sample_idxs_1) # (sample_per_class,)
        sample_idxs_2 = torch.cat(sample_idxs_2) # (sample_per_class,)
        # enumeration
        pair_idxs = pair_enumeration(sample_idxs_1.unsqueeze(1), sample_idxs_2.unsqueeze(1)).transpose(0, 1) # (2, sample_size)
        if shuffle:
            shuffle_idx = torch.arange(pair_idxs.shape[1]).tolist()
            random.shuffle(shuffle_idx)
            sample_idxs_1 = pair_idxs[0][shuffle_idx]
            sample_idxs_2 = pair_idxs[1][shuffle_idx]
        else:
            sample_idxs_1 = pair_idxs[0]
            sample_idxs_2 = pair_idxs[1]
        # # x_pair = torch.cat([x[sample_idxs_1], x[sample_idxs_2]], dim=1) # need to calculate gradient for this step
        # # y_pair = (y[sample_idxs_1] == y[sample_idxs_2]).long()
        return sample_idxs_1, sample_idxs_2

# class Pair_Enumerator:
#     # random&balance generate pair indexes
#     def __init__(self, data, mode='train'):
#         super(Pair_Enumerator, self).__init__()
#         self.num_classes = data.y.max().item() + 1 # y start from 0, -1 denotes missing
#         self.class_bucket = {} # class->indexes
#         idx2lbl = torch.Tensor(list(enumerate(data.y.tolist()))).long().transpose(0, 1) # tmp variable
#         self.mode = mode
#         for lbl in range(self.num_classes):
#             if mode == 'train':
#                 self.class_bucket[lbl] = idx2lbl[0][(idx2lbl[1] == lbl) * data.train_mask.cpu()]
#             elif mode == 'val':
#                 self.class_bucket[lbl] = idx2lbl[0][(idx2lbl[1] == lbl) * data.val_mask.cpu()]
#             elif mode == 'test':
#                 self.class_bucket[lbl] = idx2lbl[0][(idx2lbl[1] == lbl) * data.test_mask.cpu()]
#             elif mode == 'all':
#                 mask_all = (data.train_mask + data.val_mask + data.test_mask).cpu()
#                 self.class_bucket[lbl] = idx2lbl[0][idx2lbl[1] == lbl * mask_all]
#             else:
#                 raise NotImplementedError('Not Implemented Mode:{}'.format(mode))
    
#     def sampling(self, max_class_num=2, sample_size=10000, shuffle=True):
#         # x may be raw_feature/hidden_feature
#         # sample max_class_num among all classes from which to conduct sampling, to keep label-balance
        
#         if self.num_classes > max_class_num:
#             selected_classes = np.random.choice(torch.arange(self.num_classes), replace=False, size=max_class_num) # sampling classes without putback
#         else:
#             selected_classes = np.arange(self.num_classes).astype(np.int8)
#         sample_idxs_1 = []
#         sample_idxs_2 = []
#         sample_per_class = int(np.sqrt(sample_size) / max_class_num)
#         for lbl in selected_classes:
#             sample_idxs_1.append(torch.from_numpy(np.random.choice(self.class_bucket[lbl], size=sample_per_class)).long()) # sampling with putback
#             sample_idxs_2.append(torch.from_numpy(np.random.choice(self.class_bucket[lbl], size=sample_per_class)).long()) # sampling with putback
#         sample_idxs_1 = torch.cat(sample_idxs_1) # (sample_per_class,)
#         sample_idxs_2 = torch.cat(sample_idxs_2) # (sample_per_class,)
#         # enumeration
#         pair_idxs = pair_enumeration(sample_idxs_1.unsqueeze(1), sample_idxs_2.unsqueeze(1)).transpose(0, 1) # (2, sample_size)
#         if shuffle:
#             shuffle_idx = torch.arange(pair_idxs.shape[1]).tolist()
#             random.shuffle(shuffle_idx)
#             sample_idxs_1 = pair_idxs[0][shuffle_idx]
#             sample_idxs_2 = pair_idxs[1][shuffle_idx]
#         else:
#             sample_idxs_1 = pair_idxs[0]
#             sample_idxs_2 = pair_idxs[1]
#         # # x_pair = torch.cat([x[sample_idxs_1], x[sample_idxs_2]], dim=1) # need to calculate gradient for this step
#         # # y_pair = (y[sample_idxs_1] == y[sample_idxs_2]).long()
#         return sample_idxs_1, sample_idxs_2

class Pair_Enumerator:
    # random&balance generate pair indexes
    def __init__(self, data, mode='train'):
        super(Pair_Enumerator, self).__init__()
        self.num_classes = data.y.max().item() + 1 # y start from 0, -1 denotes missing
        self.class_bucket = {} # class->indexes
        idx2lbl = torch.Tensor(list(enumerate(data.y.tolist()))).long().transpose(0, 1) # tmp variable
        self.mode = mode
        for lbl in range(self.num_classes):
            if mode == 'train':
                self.class_bucket[lbl] = idx2lbl[0][(idx2lbl[1] == lbl) * data.train_mask.cpu()]
            elif mode == 'val':
                self.class_bucket[lbl] = idx2lbl[0][(idx2lbl[1] == lbl) * data.val_mask.cpu()]
            elif mode == 'test':
                self.class_bucket[lbl] = idx2lbl[0][(idx2lbl[1] == lbl) * data.test_mask.cpu()]
            elif mode == 'all':
                mask_all = (data.train_mask + data.val_mask + data.test_mask).cpu()
                self.class_bucket[lbl] = idx2lbl[0][idx2lbl[1] == lbl * mask_all]
            else:
                raise NotImplementedError('Not Implemented Mode:{}'.format(mode))
    def balanced_sampling(self, max_class_num=2, sample_size=10000, shuffle=True):
        # x may be raw_feature/hidden_feature
        # sample max_class_num among all classes from which to conduct sampling, to keep label-balance
        
        if self.num_classes > max_class_num:
            selected_classes = np.random.choice(torch.arange(self.num_classes), replace=False, size=max_class_num) # sampling classes without putback
        else:
            selected_classes = np.arange(self.num_classes).astype(np.int8)
        # same-class pair generation
        sample_idxs_1 = []
        sample_idxs_2 = []
        sample_per_class_same = int(0.5 * sample_size / max_class_num)
        sample_per_class_diff = int(0.5 * sample_size / (max_class_num * (max_class_num - 1)))
        for lbl_1 in selected_classes:
            for lbl_2 in selected_classes:
                if lbl_1 == lbl_2:
                    idx1 = torch.from_numpy(np.random.choice(self.class_bucket[lbl_1], size=sample_per_class_same)).long() # sampling with putback
                    idx2 = torch.from_numpy(np.random.choice(self.class_bucket[lbl_2], size=sample_per_class_same)).long() # sampling with putback
                else:
                    idx1 = torch.from_numpy(np.random.choice(self.class_bucket[lbl_1], size=sample_per_class_diff)).long() # sampling with putback
                    idx2 = torch.from_numpy(np.random.choice(self.class_bucket[lbl_2], size=sample_per_class_diff)).long() # sampling with putback
                # pair_idxs = torch.stack((idx1, idx2), dim=0) # 2 * pair_num
                sample_idxs_1.append(idx1) 
                sample_idxs_2.append(idx2)
        sample_idxs_1 = torch.cat(sample_idxs_1, dim=0)
        sample_idxs_2 = torch.cat(sample_idxs_2, dim=0)

        # shuffle
        if shuffle:
            shuffle_idx = torch.arange(sample_idxs_1.shape[0]).tolist()
            random.shuffle(shuffle_idx)
            sample_idxs_1 = sample_idxs_1[shuffle_idx]
            sample_idxs_2 = sample_idxs_1[shuffle_idx]
        # # x_pair = torch.cat([x[sample_idxs_1], x[sample_idxs_2]], dim=1) # need to calculate gradient for this step
        # # y_pair = (y[sample_idxs_1] == y[sample_idxs_2]).long()
        return sample_idxs_1, sample_idxs_2
    def sampling(self, max_class_num=2, sample_size=10000, shuffle=True):
        # x may be raw_feature/hidden_feature
        # sample max_class_num among all classes from which to conduct sampling, to keep label-balance
        
        if self.num_classes > max_class_num:
            selected_classes = np.random.choice(torch.arange(self.num_classes), replace=False, size=max_class_num) # sampling classes without putback
        else:
            selected_classes = np.arange(self.num_classes).astype(np.int8)
        sample_idxs_1 = []
        sample_idxs_2 = []
        sample_per_class = int(np.sqrt(sample_size) / max_class_num)
        for lbl in selected_classes:
            sample_idxs_1.append(torch.from_numpy(np.random.choice(self.class_bucket[lbl], size=sample_per_class)).long()) # sampling with putback
            sample_idxs_2.append(torch.from_numpy(np.random.choice(self.class_bucket[lbl], size=sample_per_class)).long()) # sampling with putback
        sample_idxs_1 = torch.cat(sample_idxs_1) # (sample_per_class,)
        sample_idxs_2 = torch.cat(sample_idxs_2) # (sample_per_class,)
        # enumeration
        pair_idxs = pair_enumeration(sample_idxs_1.unsqueeze(1), sample_idxs_2.unsqueeze(1)).transpose(0, 1) # (2, sample_size)
        if shuffle:
            shuffle_idx = torch.arange(pair_idxs.shape[1]).tolist()
            random.shuffle(shuffle_idx)
            sample_idxs_1 = pair_idxs[0][shuffle_idx]
            sample_idxs_2 = pair_idxs[1][shuffle_idx]
        else:
            sample_idxs_1 = pair_idxs[0]
            sample_idxs_2 = pair_idxs[1]
        # # x_pair = torch.cat([x[sample_idxs_1], x[sample_idxs_2]], dim=1) # need to calculate gradient for this step
        # # y_pair = (y[sample_idxs_1] == y[sample_idxs_2]).long()
        return sample_idxs_1, sample_idxs_2

def generate_pairs(x, y, idx1, idx2):
    x_pair = torch.cat([x[idx1], x[idx2]], dim=1) # need to calculate gradient for this step
    y_pair = (y[idx1] == y[idx2]).long()
    return x_pair, y_pair


class SimNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layer=2, use_bn=False, dropout=0.5, act_fn='relu'):
        super(SimNet, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.use_bn = use_bn
        self.dropout = dropout
        self.num_layer = num_layer

        # linear layers
        if num_layer == 1:
            self.layers.append(Linear(dim_in*2, 1, bias=True)) # input layer
        else:
            self.layers.append(Linear(dim_in*2, dim_hidden, bias=True)) # input layer
            for _ in range(num_layer-2):
                self.layers.append(Linear(dim_hidden, dim_hidden, bias=True))
            self.layers.append(Linear(dim_hidden, 1, bias=True)) # input layer

        # bn
        if use_bn:
            self.bns = torch.nn.ModuleList()
            for _ in range(num_layer-1):
                self.bns.append(nn.BatchNorm1d(dim_hidden))
                
        # activation function
        self.act_fn = act_fn
        if act_fn == 'relu':
            self.act_fn = nn.ReLU()
        elif act_fn == 'leakyrelu':
            self.act_fn = nn.LeakyReLU(0.2, inplace=False)
        elif act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        elif act_fn == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        else:
            raise NotImplementedError('Not Implemented Activation Function:{}'.format(act_fn))
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, z, idx1, idx2):
        x = torch.cat((z[idx1], z[idx2]), dim=1)
        for idx in range(self.num_layer-1):
            x = self.layers[idx](x)
            if self.use_bn:
                x = self.bns[idx](x)
            x = self.act_fn(x)
        logits = self.layers[-1](x)
        similarity = torch.sigmoid(logits)
        return similarity

class Source_Learner(torch.nn.Module):
    
    def __init__(self, data, dim_hidden=64, norm_mode='None', norm_scale=1, use_clf=True):
        super(Source_Learner, self).__init__()
        self.dim_in = data.num_features
        self.num_classes = data.y.max().item() + 1
        self.dim_hidden = dim_hidden
        self.backbone = GraphEncoder(self.dim_in, self.dim_hidden, dim_hidden=self.dim_hidden, \
            layer_num=2, root_weight=True, norm_mode=norm_mode, norm_scale=norm_scale, log_softmax=False)
        # self.pair_enumerator = Pair_Enumerator(data)

        self.sim_net = Similar(self.dim_hidden, num_clf_classes=data.y.max().item()+1, dropout=0.6, use_clf=use_clf)
        # self.sim_net = Similar_noTrans(self.dim_hidden, num_clf_classes=data.y.max().item()+1, dropout=0.6, use_clf=use_clf)

        # self.sim_net = SimNet(self.dim_hidden, self.dim_hidden, num_layer=1, \
        #     use_bn=False, dropout = 0.5, act_fn = 'relu') # for debug
        # self.sim_net = SimNet(self.dim_hidden, self.dim_hidden, num_layer=2, \
        #     use_bn=False, dropout = 0.5, act_fn = 'relu')

        # self.sim_net = nn.Sequential(
        #     # Linear(dim_hidden*2, 1),
        #     Linear(dim_hidden, 1), # for debug
        #     nn.Sigmoid()
        # )
        # self.idx1, self.idx2 = self.pair_enumerator.sampling(max_class_num=2, sample_size=10000, shuffle=False)
        self.reset_parameters()
    def reset_parameters(self):
        self.backbone.reset_parameters()
        self.sim_net.reset_parameters()
    def forward(self, data, idx1, idx2, return_representation=False):
        h = self.backbone(data.x, data.edge_index)
        # h = data.x
        # idx1, idx2 = self.pair_enumerator.sampling(max_class_num=max_class_num, sample_size=sample_size, shuffle=False)
        # idx1, idx2 = self.idx1, self.idx2
        # print(h.grad)
        # h_pair, y_pair = generate_pairs(h, data.y, idx1, idx2)
        # h_pair = torch.cat((h[idx1], h[idx2]), dim=1) # need to calculate gradient for this step

        # y_pair = (data.y[idx1] == data.y[idx2]).float()
        # h_pair, y_pair = h[data.train_mask], data.y[data.train_mask] # for debug
        probs_pair, logits_clf = self.sim_net(h, idx1, idx2)
        # print(self.sim_net[0].weight.grad)
        if return_representation:
            return probs_pair, logits_clf, h
        else:
            return probs_pair, logits_clf
        # return probs_pair, y_pair

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
       
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        h = torch.mm(input, self.weight)
        output = torch.spmm(adj, h)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->{})".format(
                    self.in_features, self.out_features)

    
class Decoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layer=2, use_norm=False, dropout=0.5, act_fn='relu', norm_mode='PN', norm_scale=1.):
        super(Decoder, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.use_norm = use_norm
        self.dropout = dropout
        self.num_layer = num_layer

        # linear layers
        if num_layer == 1:
            self.layers.append(Linear(dim_in, dim_out, bias=True)) # input layer
        else:
            self.layers.append(Linear(dim_in, dim_hidden, bias=True)) # input layer
            for _ in range(num_layer-2):
                self.layers.append(Linear(dim_hidden, dim_hidden, bias=True))
            self.layers.append(Linear(dim_hidden, dim_out, bias=True)) # input layer

        # bn
        if use_norm:
            self.pair_norm = PairNorm(norm_mode, norm_scale)
                
        # activation function
        self.act_fn = act_fn
        if act_fn == 'relu':
            self.act_fn = nn.ReLU()
        elif act_fn == 'leakyrelu':
            self.act_fn = nn.LeakyReLU(0.2, inplace=False)
        elif act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        elif act_fn == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        else:
            raise NotImplementedError('Not Implemented Activation Function:{}'.format(act_fn))
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def forward(self, z):
        x = z
        for idx in range(self.num_layer-1):
            x = self.layers[idx](x)
            if self.use_norm:
                x = self.pair_norm(x)
            x = self.act_fn(x)
        recons = self.layers[-1](x)
        return recons


class Target_Learner_AE(torch.nn.Module):
    
    def __init__(self, data, dim_eq_trans=128, dim_hidden=64, norm_mode='None', norm_scale=1):
        super(Target_Learner_AE, self).__init__()
        self.dim_in = data.num_features
        self.dim_eq_trans = dim_eq_trans
        self.num_classes = data.y.max().item() + 1
        self.dim_hidden = dim_hidden
        self.equavilent_trans_layer = nn.Sequential(
            Linear(self.dim_in, dim_eq_trans, bias=True),
            PairNorm(mode=norm_mode, scale=norm_scale),
            nn.Tanh()
        )

        self.encoder = GraphEncoder(dim_eq_trans, dim_hidden, dim_hidden=dim_hidden, \
            layer_num=2, root_weight=True, norm_mode=norm_mode, norm_scale=norm_scale, log_softmax=False)

        self.decoder = Decoder(dim_hidden, dim_hidden, dim_eq_trans, num_layer=2, \
            use_norm=True, dropout = 0.5, act_fn = 'relu', norm_mode=norm_mode, norm_scale=norm_scale)

        # self.encoder = GraphEncoder(self.dim_in, dim_hidden, dim_hidden=dim_hidden, \
        #     layer_num=2, root_weight=True, norm_mode=norm_mode, norm_scale=norm_scale, log_softmax=False)

        # self.decoder = Decoder(dim_hidden, dim_hidden, self.dim_in, num_layer=2, \
        #     use_norm=True, dropout = 0.5, act_fn = 'relu', norm_mode=norm_mode, norm_scale=norm_scale)

        self.reset_parameters()
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
    
    def encode(self, data):
        h0 = self.equavilent_trans_layer(data.x)
        # h0 = data.x
        z = self.encoder(h0, data.edge_index)
        return z, h0
        
    
    def decode(self, z):
        recons = self.decoder(z)
        recons = torch.tanh(recons)
        return recons
    
    def forward(self, data):
        z, h0 = self.encode(data)
        recons = self.decode(z)
        return h0, z, recons
    

class Discriminator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layer=2, use_bn=False, use_pair_norm=False, dropout=0.5, act_fn='leakyrelu', sigmoid_output=True, norm_mode='PN', norm_scale=1.):
        super(Discriminator, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.use_bn = use_bn
        self.dropout = dropout
        self.num_layer = num_layer
        self.sigmoid_output = sigmoid_output

        # linear layers
        if num_layer == 1:
            self.layers.append(Linear(dim_in, 1, bias=True)) # input layer
        else:
            self.layers.append(Linear(dim_in, dim_hidden, bias=True)) # input layer
            for _ in range(num_layer-2):
                self.layers.append(Linear(dim_hidden, dim_hidden, bias=True))
            self.layers.append(Linear(dim_hidden, 1, bias=True)) # input layer
        self.use_pair_norm = use_pair_norm
        if use_pair_norm:
            self.pair_norm = PairNorm(mode=norm_mode, scale=norm_scale)

        # bn
        if use_bn:
            self.bns = torch.nn.ModuleList()
            for _ in range(num_layer-1):
                self.bns.append(nn.BatchNorm1d(dim_hidden))
                
        # activation function
        self.act_fn = act_fn
        if act_fn == 'relu':
            self.act_fn = nn.ReLU()
        elif act_fn == 'leakyrelu':
            self.act_fn = nn.LeakyReLU(0.2, inplace=False)
        elif act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        elif act_fn == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        else:
            raise NotImplementedError('Not Implemented Activation Function:{}'.format(act_fn))
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, z):
        x = z
        for idx in range(self.num_layer-1):
            x = self.layers[idx](x)
            if self.use_bn:
                x = self.bns[idx](x)
            elif self.use_pair_norm:
                x = self.pair_norm(x)
            x = self.act_fn(x)
        logits = self.layers[-1](x)
        probs = torch.sigmoid(logits) if self.sigmoid_output else logits
        return probs
    
class Adversarial_Learner(nn.Module):
    def __init__(self, data_src, data_tar, dim_hidden=64, num_layer=2, source_clf=True, norm_mode='PN', norm_scale=1.):
        super(Adversarial_Learner, self).__init__()
        self.num_layer = num_layer
        self.source_clf = source_clf
        self.source_learner = Source_Learner(data_src, dim_hidden=dim_hidden, norm_mode=norm_mode, norm_scale=norm_scale, use_clf=source_clf)
        self.target_learner = Target_Learner_AE(data_tar, dim_eq_trans=128, dim_hidden=dim_hidden, norm_mode=norm_mode, norm_scale=norm_scale)
        self.discriminator = Discriminator(dim_hidden, dim_hidden, num_layer=2, use_pair_norm=False, \
            dropout=0.5, act_fn='relu', sigmoid_output=True, norm_mode=norm_mode, norm_scale=norm_scale)
    def get_probs_within_domain(self, data, idx1, idx2, domain='target'):
        # domain = 'source'/'target'
        if domain == 'source':
            probs_pair, log_probs_clf = self.source_learner(data, idx1, idx2, return_representation=False)
        elif domain == 'target':
            z, _ = self.target_learner.encode(data)
            probs_pair, log_probs_clf = self.source_learner.sim_net(z, idx1, idx2)
        if not self.source_clf:
            log_probs_clf = torch.zeros((data.x.shape[0], data.y.max().item()+1))
        return probs_pair, log_probs_clf.exp()
    def get_probs_cross_domain(self, data_src, data_tar, idx1, idx2, return_representation=False):
        z_src = self.source_learner.backbone(data_src.x, data_src.edge_index)
        z_tar, _ = self.target_learner.encode(data_tar)
        probs_pair, log_probs_clf_src, log_probs_clf_tar = self.source_learner.sim_net.forward_cross_domain(z_src, z_tar, idx1, idx2)
        if not self.source_clf:
            log_probs_clf_src = torch.zeros((z_src.shape[0], data_src.y.max().item()+1))
            log_probs_clf_tar = torch.zeros((z_tar.shape[0], data_tar.y.max().item()+1))
        if return_representation:
            return probs_pair, log_probs_clf_src.exp(), log_probs_clf_tar.exp(), z_src.detach(), z_tar.detach()
        else:
            return probs_pair, log_probs_clf_src.exp(), log_probs_clf_tar.exp()




# model v2 for non-graph dataset


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

class Similar_v2(torch.nn.Module):
    def __init__(self, in_channels, num_clf_classes, dropout=0.6, use_clf=True, mode='cosine'):
        super(Similar_v2, self).__init__()
        self.mode = mode
        if mode == 'cosine':
            self.biasatt = nn.Sequential(
                Linear(128, 64, bias=True, weight_initializer='glorot'),
                nn.Tanh(),   
                Linear(64, 128, bias=True, weight_initializer='glorot'),
            )

            for m in self.biasatt:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
            
            self.lin_self = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                Linear(in_channels, 64, bias=False, weight_initializer='glorot'),
                nn.BatchNorm1d(64),
                nn.Tanh(),  
                Linear(64, 128, bias=False, weight_initializer='glorot'),
            )
        elif mode == 'mlp':
            self.lin_self = nn.Sequential(
                nn.BatchNorm1d(in_channels * 2),
                Linear(in_channels * 2, 128, bias=True, weight_initializer='glorot'),
                nn.BatchNorm1d(128),
                nn.ReLU(),  
                Linear(128, 1, bias=True, weight_initializer='glorot'),
            )
        else:
            raise NotImplementedError('Not Supported Mode:{}'.format(mode))

        self.use_clf = use_clf
        if use_clf:
            self.lin_clf = Linear(in_channels, num_clf_classes, bias=True, weight_initializer='glorot')
        
        self.dropout = dropout
        # self.adj_t_cache = None
        self.reset_parameters()
    def reset_parameters(self):
        if self.use_clf:
            self.lin_clf.reset_parameters()
        for m in self.lin_self:
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight)
                m.reset_parameters()

    def similarity_cross_domain(self, x_src, x_tar, idx1, idx2) -> Tensor:
        if self.mode == 'cosine':
            z_src = self.lin_self(x_src)
            z_tar = self.lin_self(x_tar)
            alpha = torch.nn.CosineSimilarity(dim=1)(z_src[idx1]+ self.biasatt(z_src[idx1]), z_tar[idx2]+ self.biasatt(z_tar[idx2]))
        elif self.mode == 'mlp':
            x_pair = torch.cat((x_src[idx1], x_tar[idx2]), dim=1)
            alpha = self.lin_self(x_pair).squeeze(-1)
        # alpha = torch.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training)*5)
        alpha = torch.sigmoid(alpha)
        return alpha

    def forward_cross_domain(self, x_src, x_tar, idx1, idx2):
        z_src, z_tar = x_src, x_tar
        #classifier
        log_probs_clf_src = log_probs_clf_tar = None
        if self.use_clf:
            logits_src = self.lin_clf(F.dropout(F.relu(z_src), p=self.dropout, training=self.training))
            logits_tar = self.lin_clf(F.dropout(F.relu(z_tar), p=self.dropout, training=self.training))
            log_probs_clf_src = F.log_softmax(logits_src, dim=-1)
            log_probs_clf_tar = F.log_softmax(logits_tar, dim=-1)
        alpha = self.similarity_cross_domain(z_src, z_tar, idx1, idx2)
        return alpha.unsqueeze(-1), log_probs_clf_src, log_probs_clf_tar

    def similarity(self, x, idx1, idx2) -> Tensor:
        if self.mode == 'cosine':
            z = self.lin_self(x)
            alpha = torch.nn.CosineSimilarity(dim=1)(z[idx1]+ self.biasatt(z[idx1]), z[idx2]+ self.biasatt(z[idx2]))
        elif self.mode == 'mlp':
            x_pair = torch.cat((x[idx1], x[idx2]), dim=1)
            alpha = self.lin_self(x_pair).squeeze(-1)
        # alpha = torch.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training)*5)
        alpha = torch.sigmoid(alpha)
        return alpha

    def forward(self, x, idx1, idx2):
        # x, edge_index = data.x, data.edge_index
        
        # assert isinstance(edge_index, torch.Tensor)
        # if  self.adj_t_cache == None:
        #     self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))

        # adj_t=self.adj_t_cache
        # z = self.lin_input(x)
        z = x
        # mu_neb = matmul(adj_t, z, reduce='mean')

        #classifier
        log_probs_clf = None
        if self.use_clf:
            logits = self.lin_clf(F.dropout(F.relu(z), p=self.dropout, training=self.training))
            log_probs_clf = F.log_softmax(logits, dim=-1)
        alpha = self.similarity(z, idx1, idx2)
        return alpha.unsqueeze(-1), log_probs_clf

class Source_Learner_v2(torch.nn.Module):
    
    def __init__(self, data, dim_hidden=64, norm_mode='None', norm_scale=1, use_clf=True, use_norm=True, backbone='mlp', mode='cosine'):
        super(Source_Learner_v2, self).__init__()
        self.dim_in = data.num_features
        self.num_classes = data.y.max().item() + 1
        self.dim_hidden = dim_hidden
        if backbone == 'gnn':
            self.backbone = GraphEncoder(self.dim_in, self.dim_hidden, dim_hidden=self.dim_hidden, \
                layer_num=2, root_weight=True, use_norm=use_norm, norm_mode=norm_mode, \
                    norm_scale=norm_scale, log_softmax=False)
        elif backbone == 'mlp':
            self.backbone = MLP(self.dim_in, self.dim_hidden, dim_hidden=self.dim_hidden, \
                layer_num=2, root_weight=True, use_norm=use_norm, norm_mode=norm_mode, \
                    norm_scale=norm_scale, log_softmax=False)
        else:
            raise NotImplementedError('Not Implemented Backbone:{}'.format(backbone))
        # self.pair_enumerator = Pair_Enumerator(data)

        self.sim_net = Similar_v2(self.dim_hidden, num_clf_classes=data.y.max().item()+1, dropout=0.6, use_clf=use_clf, mode=mode)
        # self.sim_net = Similar_noTrans(self.dim_hidden, num_clf_classes=data.y.max().item()+1, dropout=0.6, use_clf=use_clf)

        # self.sim_net = SimNet(self.dim_hidden, self.dim_hidden, num_layer=1, \
        #     use_bn=False, dropout = 0.5, act_fn = 'relu') # for debug
        # self.sim_net = SimNet(self.dim_hidden, self.dim_hidden, num_layer=2, \
        #     use_bn=False, dropout = 0.5, act_fn = 'relu')

        # self.sim_net = nn.Sequential(
        #     # Linear(dim_hidden*2, 1),
        #     Linear(dim_hidden, 1), # for debug
        #     nn.Sigmoid()
        # )
        self.reset_parameters()
    def reset_parameters(self):
        self.backbone.reset_parameters()
        self.sim_net.reset_parameters()
    def forward(self, data, idx1, idx2, return_representation=False):
        h = self.backbone(data.x, data.edge_index)
        # h = data.x
        # idx1, idx2 = self.pair_enumerator.sampling(max_class_num=max_class_num, sample_size=sample_size, shuffle=False)
        # idx1, idx2 = self.idx1, self.idx2
        # print(h.grad)
        # h_pair, y_pair = generate_pairs(h, data.y, idx1, idx2)
        # h_pair = torch.cat((h[idx1], h[idx2]), dim=1) # need to calculate gradient for this step

        # y_pair = (data.y[idx1] == data.y[idx2]).float()
        # h_pair, y_pair = h[data.train_mask], data.y[data.train_mask] # for debug
        probs_pair, logits_clf = self.sim_net(h, idx1, idx2)
        # print(self.sim_net[0].weight.grad)
        if return_representation:
            return probs_pair, logits_clf, h
        else:
            return probs_pair, logits_clf
        # return probs_pair, y_pair


class Target_Learner_AE_v2(torch.nn.Module):
    
    def __init__(self, data, dim_eq_trans=128, dim_hidden=64, use_norm=True, norm_mode='None', norm_scale=1, backbone='mlp'):
        super(Target_Learner_AE_v2, self).__init__()
        self.dim_in = data.num_features
        self.dim_eq_trans = dim_eq_trans
        self.num_classes = data.y.max().item() + 1
        self.dim_hidden = dim_hidden
        self.equavilent_trans_layer = nn.Sequential(
            Linear(self.dim_in, dim_eq_trans, bias=True),
            PairNorm(mode=norm_mode, scale=norm_scale),
            nn.Tanh()
        )
        self.use_norm = use_norm
        if backbone == 'gnn':
            self.encoder = GraphEncoder(dim_eq_trans, dim_hidden, dim_hidden=dim_hidden, \
                layer_num=2, root_weight=True, use_norm=use_norm, norm_mode=norm_mode, norm_scale=norm_scale, log_softmax=False)
        elif backbone == 'mlp':
            self.encoder = MLP(dim_eq_trans, dim_hidden, dim_hidden=dim_hidden, \
                layer_num=2, root_weight=True, use_norm=use_norm, norm_mode=norm_mode, norm_scale=norm_scale, log_softmax=False)
        else:
            raise NotImplementedError('Not Implemented Backbone:{}'.format(backbone))

        self.decoder = Decoder(dim_hidden, dim_hidden, dim_eq_trans, num_layer=2, \
            use_norm=True, dropout = 0.5, act_fn = 'relu', norm_mode=norm_mode, norm_scale=norm_scale)

        # self.encoder = GraphEncoder(self.dim_in, dim_hidden, dim_hidden=dim_hidden, \
        #     layer_num=2, root_weight=True, norm_mode=norm_mode, norm_scale=norm_scale, log_softmax=False)

        # self.decoder = Decoder(dim_hidden, dim_hidden, self.dim_in, num_layer=2, \
        #     use_norm=True, dropout = 0.5, act_fn = 'relu', norm_mode=norm_mode, norm_scale=norm_scale)

        self.reset_parameters()
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
    
    def encode(self, data):
        h0 = self.equavilent_trans_layer(data.x)
        # h0 = data.x
        z = self.encoder(h0, data.edge_index)
        return z, h0
        
    
    def decode(self, z):
        recons = self.decoder(z)
        recons = torch.tanh(recons)
        return recons
    
    def forward(self, data):
        z, h0 = self.encode(data)
        recons = self.decode(z)
        return h0, z, recons


class Adversarial_Learner_v2(nn.Module):
    def __init__(self, data_src, data_tar, dim_hidden=64, num_layer=2, source_clf=True, use_norm=True, norm_mode='PN', \
                 norm_scale=1., backbone='mlp', sim_mode='cosine'):
        super(Adversarial_Learner_v2, self).__init__()
        self.num_layer = num_layer
        self.source_clf = source_clf
        self.source_learner = Source_Learner_v2(data_src, dim_hidden=dim_hidden, norm_mode=norm_mode, norm_scale=norm_scale, \
                                             use_clf=source_clf, use_norm=use_norm, backbone=backbone, mode=sim_mode)
        self.target_learner = Target_Learner_AE_v2(data_tar, dim_eq_trans=128, dim_hidden=dim_hidden, norm_mode=norm_mode, \
                                                use_norm=use_norm, norm_scale=norm_scale, backbone=backbone)
        self.discriminator = Discriminator(dim_hidden, dim_hidden, num_layer=2, use_pair_norm=False, \
            dropout=0.5, act_fn='relu', sigmoid_output=True, norm_mode=norm_mode, norm_scale=norm_scale)
    def get_probs_within_domain(self, data, idx1, idx2, domain='target'):
        # domain = 'source'/'target'
        if domain == 'source':
            probs_pair, log_probs_clf = self.source_learner(data, idx1, idx2, return_representation=False)
        elif domain == 'target':
            z, _ = self.target_learner.encode(data)
            probs_pair, log_probs_clf = self.source_learner.sim_net(z, idx1, idx2)
        if not self.source_clf:
            log_probs_clf = torch.zeros((data.x.shape[0], data.y.max().item()+1))
        return probs_pair, log_probs_clf.exp()
    def get_probs_cross_domain(self, data_src, data_tar, idx1, idx2, return_representation=False):
        z_src = self.source_learner.backbone(data_src.x, data_src.edge_index)
        z_tar, _ = self.target_learner.encode(data_tar)
        probs_pair, log_probs_clf_src, log_probs_clf_tar = self.source_learner.sim_net.forward_cross_domain(z_src, z_tar, idx1, idx2)
        if not self.source_clf:
            log_probs_clf_src = torch.zeros((z_src.shape[0], data_src.y.max().item()+1))
            log_probs_clf_tar = torch.zeros((z_tar.shape[0], data_tar.y.max().item()+1))
        if return_representation:
            return probs_pair, log_probs_clf_src.exp(), log_probs_clf_tar.exp(), z_src.detach(), z_tar.detach()
        else:
            return probs_pair, log_probs_clf_src.exp(), log_probs_clf_tar.exp()


