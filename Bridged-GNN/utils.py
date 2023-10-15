import random
import numpy as np
import torch
import torch_geometric
import torch_sparse
import torch_sparse.matmul as matmul
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def dataset_split(data, num_classes, train_val_test_ratio):
    N = data.x.shape[0]
    num_labeled = (data.y != -1).sum().item()
    print('num labeld samples:', num_labeled)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        num_class = len(idx)
        num_train_per_class = int(np.ceil(num_class * train_val_test_ratio[0]))
        num_val_per_class = int(np.floor(num_class * train_val_test_ratio[1]))
        num_test_per_class = num_class - num_train_per_class - num_val_per_class
        print('[Class:{}] Train:{} | Val:{} | Test:{}'.format(c, num_train_per_class, num_val_per_class, num_test_per_class))
        assert num_test_per_class >= 0
        idx_perm = torch.randperm(idx.size(0))
        idx_train = idx[idx_perm[:num_train_per_class]]
        idx_val = idx[idx_perm[num_train_per_class:num_train_per_class+num_val_per_class]]
        idx_test = idx[idx_perm[num_train_per_class+num_val_per_class:]]
        data.train_mask[idx_train] = True
        data.val_mask[idx_val] = True
        data.test_mask[idx_test] = True


def dataset_conversion(data, seed=0, train_val_test_ratio=[0.6,0.2,0.2], dataset_name=None, split_data=True):
    # Extract the two graph components from the original VS-Graph
    set_random_seed(seed)
    # assert dataset_name in ('company', 'twitter')
    if dataset_name in ['company', 'twitter']:
        dim_x_o = 33 if dataset_name == 'company' else 300 # 33 for company, 300 for twitter
        # feature
        x_src = data.x[data.central_mask, :]
        x_tar = data.x[~data.central_mask, :dim_x_o]
    else:
        x_src = data.x[data.central_mask]
        x_tar = data.x[~data.central_mask]
    # x_tar = data.x[~data.central_mask, :]
    print(x_src.shape, x_tar.shape)
    # edge_index
    idxs_src = torch.where(data.central_mask)[0]
    idxs_tar = torch.where(~data.central_mask)[0]
    print(idxs_src.shape[0], idxs_tar.shape[0], data.num_nodes)
    mapper_idx_src = {}
    for new_idx, ori_idx in enumerate(idxs_src):
        mapper_idx_src[ori_idx.item()] = new_idx
    mapper_idx_tar = {}
    for new_idx, ori_idx in enumerate(idxs_tar):
        mapper_idx_tar[ori_idx.item()] = new_idx
    edge_index, central_mask = data.edge_index, data.central_mask
    edge_index_src_ori = edge_index[:, central_mask[edge_index[0]] * central_mask[edge_index[1]]].tolist()
    edge_index_tar_ori = edge_index[:, ~central_mask[edge_index[0]] * ~central_mask[edge_index[1]]].tolist()
    edge_index_src = torch.LongTensor([[mapper_idx_src[idx] for idx in edge_index_src_ori[0]], [mapper_idx_src[idx] for idx in edge_index_src_ori[1]]])
    edge_index_tar = torch.LongTensor([[mapper_idx_tar[idx] for idx in edge_index_tar_ori[0]], [mapper_idx_tar[idx] for idx in edge_index_tar_ori[1]]])
    print(edge_index_src.shape, edge_index_tar.shape)
    # label
    y_src = data.y[central_mask]
    y_tar = data.y[~central_mask]
    print(y_src.shape, y_tar.shape)
    # merge
    data_src = torch_geometric.data.Data(x=x_src, edge_index=edge_index_src, y=y_src, \
        train_mask=torch.zeros(len(idxs_src)).bool(), val_mask=torch.zeros(len(idxs_src)).bool(), test_mask=torch.zeros(len(idxs_src)).bool())
    data_tar = torch_geometric.data.Data(x=x_tar, edge_index=edge_index_tar, y=y_tar, \
        train_mask=torch.zeros(len(idxs_tar)).bool(), val_mask=torch.zeros(len(idxs_tar)).bool(), test_mask=torch.zeros(len(idxs_tar)).bool())
    # dataset_split
    num_classes = data.y.max().item() + 1
    dataset_split(data_src, num_classes=num_classes, train_val_test_ratio=train_val_test_ratio)
    if split_data:
        dataset_split(data_tar, num_classes=num_classes, train_val_test_ratio=train_val_test_ratio)
    else:
        # use original datasplit in the data
        def map_func(idxs, mapper):
            for i in range(idxs.shape[0]):
                idxs[i] = mapper[idxs[i].item()]
            return idxs
        idxs_train_tar = map_func(torch.where(data.train_mask * ~data.central_mask)[0], mapper_idx_tar)
        idxs_val_tar = map_func(torch.where(data.val_mask * ~data.central_mask)[0], mapper_idx_tar)
        idxs_test_tar = map_func(torch.where(data.test_mask * ~data.central_mask)[0], mapper_idx_tar)
        data_tar.train_mask[idxs_train_tar] = True
        data_tar.val_mask[idxs_val_tar] = True
        data_tar.test_mask[idxs_test_tar] = True

    print('Dataset Conversion Done.')
    return data_src, data_tar, mapper_idx_src, mapper_idx_tar

def eval_bridged_Graph(data_merge):
    x, edge_index = data_merge.x, data_merge.edge_index
    # x, edge_index = data.x, data.edge_index
    adj_t = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], value=torch.ones(edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
    y_onehot = F.one_hot(data_merge.y + 1).cpu().float()[:, 1:] # 忽略没有标签的邻居
    lbl_dist = matmul(adj_t, y_onehot, reduce='sum')
    deg = lbl_dist.sum(1)
    mask_deg_nonzero = (lbl_dist.sum(1) != 0) * (data_merge.y.cpu() != -1)
    deg[~mask_deg_nonzero] += 1e-3
    local_homophily = (lbl_dist * y_onehot).sum(1) / deg
    avg_local_homo_ratio = (local_homophily[data_merge.test_mask] > 0.5).sum() / data_merge.test_mask.sum()
    print(avg_local_homo_ratio)
    return avg_local_homo_ratio

def eval_homophily(data):
    x, edge_index = data.x, data.edge_index
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    adj_sp = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
    # print(adj_sp)
    edge_index_2rd = torch_sparse.matmul(adj_sp, adj_sp.to_dense())
    edge_index_2rd = torch.nonzero(edge_index_2rd, as_tuple=False)
    edge_index_2rd = edge_index_2rd.transpose(0, 1)
    adj_sp_2rd = torch_sparse.SparseTensor(row=edge_index_2rd[0], col=edge_index_2rd[1], value=torch.ones(edge_index_2rd.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
    # print(adj_sp_2rd)
    mask_labled_edge_1st = (data.y[edge_index[0]] != -1) * (data.y[edge_index[1]] != -1)
    homo_ratio_1st = ((data.y[edge_index[0]] == data.y[edge_index[1]]) * mask_labled_edge_1st).sum() / mask_labled_edge_1st.sum()
    mask_labled_edge_2rd = (data.y[edge_index_2rd[0]] != -1) * (data.y[edge_index_2rd[1]] != -1)
    homo_ratio_2rd = ((data.y[edge_index_2rd[0]] == data.y[edge_index_2rd[1]]) * mask_labled_edge_2rd).sum() / mask_labled_edge_2rd.sum()
    print('homophily ratio:', homo_ratio_1st.item())
    print('homophily ratio 2rd neibors:', homo_ratio_2rd.item())

