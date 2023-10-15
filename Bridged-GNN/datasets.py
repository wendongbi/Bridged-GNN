# my own pyg dataset class
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url
import shutil
import os
import numpy as np
from utils import *
import sys
sys.path.append('./datasets')
from dataset_ktgnn import build_dataset
from torch_geometric.transforms import ToUndirected
from dataset_facebook100 import Facebook100, onehot_encoder
from collections import Counter

def prepare_datasets(dataset_name='twitter_unrelational'):
    if dataset_name == 'twitter_unrelational':
        dataset = build_dataset('twitter', split='random', split_ratio=[0.6,0.2,0.2], remove_unobserved_feats=True)
        data = dataset[0]
        # ToUndirected(merge=True)(data), data
        # data.edge_index = torch_geometric.transforms.add_self_loops.add_self_loops(data.edge_index)[0] # add self loop
        data.edge_index = torch.stack((torch.arange(data.num_nodes), torch.arange(data.num_nodes)), dim=0)
        split_data = True
    elif dataset_name == 'twitter_relational_intra_inter':
        dataset = build_dataset('twitter', split='random', split_ratio=[0.6,0.2,0.2], remove_unobserved_feats=True)
        data = dataset[0]
        ToUndirected(merge=True)(data), data
        data.edge_index = torch_geometric.transforms.add_self_loops.add_self_loops(data.edge_index)[0] # add self loop
        split_data = True
    elif dataset_name == 'office_amazon2dslr':
        path = '../datasets/office_amazon2dslr_pyg.dat'
        data = torch.load(path)
        split_data = False
    elif dataset_name == 'office_amazon2webcam':
        path = '../datasets/office_amazon2webcam_pyg.dat'
        data = torch.load(path)
        split_data = False
    elif dataset_name == 'fb_hamilton2caltech':
        path = '../datasets/dataset_FB(Hamilton->Caltech)_pyg_relational_intra.dat'
        data = torch.load(path)
        data.central_mask = data.source_mask
        del data.source_mask
        split_data = False
    elif dataset_name == 'fb_howard2simmons':
        path = '../datasets/dataset_FB(Howard->Simmons)_pyg_relational_intra.dat'
        data = torch.load(path)
        data.central_mask = data.source_mask
        del data.source_mask
        split_data = False
    # elif dataset_name == 'sync_unrelational':
    #     path = './datasets/dataset_pyg_paper/dataset_sync_pyg_unrelational.dat'
    # elif dataset_name == 'sync_relational_intra':
    #     path = './datasets/dataset_pyg_paper/dataset_sync_pyg_relational_intra.dat'
    # elif dataset_name == 'sync_relational_intra_inter':
    #     path = './datasets/dataset_pyg_paper/dataset_sync_pyg_relational_intra&inter.dat'
    else:
        raise NotImplementedError('Not Recognized Dataset Name:{}'.format(dataset_name))
    
    # data = torch.load(path)

    if dataset_name.split('_')[-1] == 'unrelational':
       data.edge_index = torch.stack((torch.arange(data.num_nodes), torch.arange(data.num_nodes)), dim=0)
    
    data_src, data_tar, mapper_idx_src, mapper_idx_tar = dataset_conversion(data, seed=1, dataset_name=dataset_name, split_data=split_data)

    return data_src, data_tar, data, mapper_idx_src, mapper_idx_tar


def Facebook100_KT(source_dataset_name, target_dataset_name, to_onehot=False, \
                   split_ratio_src=[0.4,0.3,0.3], split_ratio_tar=[0.2,0.4,0.4], seed=0, \
                    to_undirected=False, add_self_loop=False, min_sample_per_cls=150):
    set_random_seed(seed)
    data_dir = '../data/Facebook100_pyg/'
    # source dataset
    path_src = os.path.join(data_dir, source_dataset_name)
    datset_src = Facebook100(path_src, source_dataset_name, transform=None, split='random', \
            num_train_per_class=200, to_onehot=False, train_val_test_ratio=split_ratio_src)
    data_src = datset_src[0]
    # target dataset
    path_tar = os.path.join(data_dir, target_dataset_name)
    dataset_tar = Facebook100(path_tar, target_dataset_name, transform=None, split='random', \
            to_onehot=False, train_val_test_ratio=split_ratio_tar)
    data_tar = dataset_tar[0]
    # remove the classes with too few samples
    num_classes = max(datset_src.num_classes, dataset_tar.num_classes)
    N_src = data_src.num_nodes
    N_tar = data_tar.num_nodes
    mask_src_remove = torch.zeros(N_src).bool()
    mask_tar_remove = torch.zeros(N_tar).bool()
    cls_counter_src = Counter(data_src.y.view(-1).numpy())
    cls_counter_tar = Counter(data_tar.y.view(-1).numpy())
    print('[Ori] source data class dist:', sorted(dict(cls_counter_src).items(), key=lambda x:x[0]))
    print('[Ori] target data class dist:', sorted(dict(cls_counter_tar).items(), key=lambda x:x[0]))
    if min_sample_per_cls > 0:
        new_lbl_mapper = {}
        for lbl in range(num_classes):
            if cls_counter_src[lbl] < min_sample_per_cls or (cls_counter_tar[lbl] < min_sample_per_cls and cls_counter_src[lbl] < min_sample_per_cls):
                mask_src_remove[data_src.y == lbl] = True
                mask_tar_remove[data_tar.y == lbl] = True
                data_src.y[data_src.y == lbl] = -1
                data_tar.y[data_tar.y == lbl] = -1
            else:
                new_lbl_mapper[lbl] = len(new_lbl_mapper)
                data_src.y[data_src.y == lbl] = new_lbl_mapper[lbl]
                data_tar.y[data_tar.y == lbl] = new_lbl_mapper[lbl]
        data_src.train_mask[mask_src_remove] = False
        data_src.val_mask[mask_src_remove] = False
        data_src.test_mask[mask_src_remove] = False
        data_tar.train_mask[mask_tar_remove] = False
        data_tar.val_mask[mask_tar_remove] = False
        data_tar.test_mask[mask_tar_remove] = False
        print('[New] source data class dist:', sorted(dict(Counter(data_src.y.view(-1).numpy())).items(), key=lambda x:x[0]))
        print('[New] target data class dist:', sorted(dict(Counter(data_tar.y.view(-1).numpy())).items(), key=lambda x:x[0]))

    if to_onehot:
        x_merge = torch.cat((data_src.x, data_tar.x), dim=0)
        x_merge_onehot = onehot_encoder(x_merge)
        data_src.x = x_merge_onehot[:N_src]
        data_tar.x = x_merge_onehot[N_src:]
    
    

    if to_undirected:
        from torch_geometric.transforms import ToUndirected
        ToUndirected(merge=True)(data_src)
        ToUndirected(merge=True)(data_tar)
    if add_self_loop:
        data_src.edge_index = T.add_self_loops.add_self_loops(data_src.edge_index)[0] # add self loop
        data_tar.edge_index = T.add_self_loops.add_self_loops(data_tar.edge_index)[0] # add self loop
    return data_src, data_tar
# source_dataset_name='Hamilton46' # Howard90, Hamilton46,
# target_dataset_name='Caltech36' # Simmons81, Caltech36

source_dataset_name='Hamilton46' # Howard90, Hamilton46,
target_dataset_name='Caltech36' # Simmons81, Caltech36
data_src, data_tar = Facebook100_KT(source_dataset_name=source_dataset_name, target_dataset_name=target_dataset_name, \
                                    to_onehot=True, split_ratio_src=[0.4,0.3,0.3], split_ratio_tar=[0.2,0.4,0.4], seed=0, \
                                        to_undirected=False, add_self_loop=False, min_sample_per_cls=50)
data_src, data_tar
    

