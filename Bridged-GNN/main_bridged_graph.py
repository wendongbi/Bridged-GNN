import torch
from datasets import prepare_datasets
from torch_geometric.nn import GATConv
import sys, os
sys.path.append('./models')
from backbones import *
import random
from utils import set_random_seed, dataset_conversion
import argparse
import torch_sparse
from torch_sparse import matmul
import torch_geometric

# import functions for model training
import random
import time
import sys
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch.autograd import Variable
import networkx as nx
import copy
sys.path.append('./models')
from models import *
import numpy as np
import torch.nn.functional as F
from utils import *
from scripts import main_adv, main_adv_v2

# adversarial_loss = torch.nn.BCELoss()
# adversarial_loss = adversarial_loss.to(device)


def add_topk_sim_cross_domain_edges(data_src, data_tar, model, epsilon=0.5, k=3, batch_size=1000):
    # Possible edge types of src-tar cross-domain edges: 
    #   * train-train; train-val; train-test;
    #   * val-val, val-test, test-test
    set_random_seed(0)
    num_src_nodes = data_src.x.shape[0]
    num_tar_nodes = data_tar.x.shape[0]
    all_idx_src = torch.arange(num_src_nodes).unsqueeze(-1)
    start_idx = 0
    edge_index_bucket = []
    e_sim_mat = []
    idx_src_mat = []
    while start_idx < num_tar_nodes:
        # left-close&right-open interval
        end_idx = min(start_idx+batch_size, num_tar_nodes)
        batch_idx_tar = torch.arange(start_idx, end_idx, step=1).unsqueeze(-1)
        pair_idxs = pair_enumeration(all_idx_src, batch_idx_tar).transpose(0, 1)
        
        idx1, idx2 = pair_idxs[0], pair_idxs[1]
        # print(idx1, idx2)
        # Do not need to remove self-loops for adding cross-domain edges, but need that step for adding within-domain edges
        with torch.no_grad():
            model.eval()
            probs_pair, probs_clf_src, probs_clf_tar, _, _ = model.get_probs_cross_domain(data_src, data_tar, idx1, idx2, return_representation=True)
            # leave double-check by probs_clf_src and probls_clf_tar as further work
            # leave using edge weighted graph as further work
            sim_mat = probs_pair.squeeze(-1).view(-1, num_src_nodes) # bs * num_src_nodes
            topk_sim = sim_mat.topk(k=k, dim=1, largest=True, sorted=False)
            topk_idx_tar = torch.cat([batch_idx_tar for _ in range(k)], dim=1).view(-1)
            topk_idx_src = topk_sim.indices.view(-1).cpu()
            edge_index_topk_batch = torch.stack((topk_idx_src, topk_idx_tar), dim=0)
            edge_index_bucket.append(edge_index_topk_batch)
            e_sim_mat.append(topk_sim.values.cpu())
            idx_src_mat.append(topk_sim.indices.cpu())
        start_idx = end_idx
    edge_index_added = torch.cat(edge_index_bucket, dim=1)
    e_sim_mat = torch.cat(e_sim_mat, dim=0)
    idx_src_mat = torch.cat(idx_src_mat, dim=0)
    mask_labeled_edges = (data_src.y[edge_index_added[0]] != -1) * (data_tar.y[edge_index_added[1]] != -1)
    new_homo_ratio = ((data_src.y[edge_index_added[0]] == data_tar.y[edge_index_added[1]]) * mask_labeled_edges).sum() / mask_labeled_edges.sum()
    new_homo_ratio = new_homo_ratio.cpu().item()
    print('Current homophily ratio:', new_homo_ratio)
    return torch_geometric.utils.coalesce(edge_index_added), e_sim_mat, idx_src_mat, probs_clf_src, probs_clf_tar

def add_topk_sim_within_domain_edges(data_src, model, k=3, batch_size=1000, domain='source'):
    # Possible edge types of src-tar cross-domain edges: 
    #   * train-train; train-val; train-test;
    #   * val-val, val-test, test-test
    set_random_seed(0)
    num_src_nodes = data_src.x.shape[0]
    all_idx_src = torch.arange(num_src_nodes).unsqueeze(-1)

    # 1. construct source-graph
    start_idx = 0
    edge_index_bucket_src = []
    e_sim_mat_src = []
    idx_src_mat_src = []
    while start_idx < num_src_nodes:
        # left-close&right-open interval
        end_idx = min(start_idx+batch_size, num_src_nodes)
        batch_idx_src = torch.arange(start_idx, end_idx, step=1).unsqueeze(-1)
        pair_idxs = pair_enumeration(all_idx_src, batch_idx_src).transpose(0, 1)
        
        idx1, idx2 = pair_idxs[0], pair_idxs[1]
        # Do not need to remove self-loops for adding cross-domain edges, but need that step for adding within-domain edges
        with torch.no_grad():
            model.eval()
            probs_pair, _ = model.get_probs_within_domain(data_src, idx1, idx2, domain=domain)
            # leave double-check by probs_clf_src and probls_clf_tar as further work
            # leave using edge weighted graph as further work
            sim_mat = probs_pair.squeeze(-1).view(-1, num_src_nodes) # bs * num_src_nodes
            topk_sim = sim_mat.topk(k=k, dim=1, largest=True, sorted=False)
            topk_idx_to = torch.cat([batch_idx_src for _ in range(k)], dim=1).view(-1)
            topk_idx_from = topk_sim.indices.view(-1).cpu()
            edge_index_topk_batch = torch.stack((topk_idx_from, topk_idx_to), dim=0)
            edge_index_bucket_src.append(edge_index_topk_batch)
            e_sim_mat_src.append(topk_sim.values.cpu())
            idx_src_mat_src.append(topk_sim.indices.cpu())
        start_idx = end_idx
    edge_index_added_src = torch.cat(edge_index_bucket_src, dim=1)
    edge_index_added_src = torch_geometric.utils.coalesce(edge_index_added_src) # remove duplicated edges
    e_sim_mat_src = torch.cat(e_sim_mat_src, dim=0)
    idx_src_mat_src = torch.cat(idx_src_mat_src, dim=0)
    mask_labeled_edges_src = (data_src.y[edge_index_added_src[0]] != -1) * (data_src.y[edge_index_added_src[1]] != -1)
    new_homo_ratio_src = ((data_src.y[edge_index_added_src[0]] == data_src.y[edge_index_added_src[1]]) * mask_labeled_edges_src).sum() / mask_labeled_edges_src.sum()
    new_homo_ratio_src = new_homo_ratio_src.cpu().item()
    print('Current homophily ratio of Graph:', new_homo_ratio_src)
    return edge_index_added_src, e_sim_mat_src, idx_src_mat_src


def check_added_edges_within_domain_validity(edge_index_added, e_sim, data_in, probs_clf, thres_conf_quantile=0.1, thres_feat_sim=0.):
    # homophiy ratio of orignal cross-domain edges
    mask_labeled_edges = (data_in.y[edge_index_added[0]] != -1) * (data_in.y[edge_index_added[1]] != -1)
    ori_homo_ratio = ((data_in.y[edge_index_added[0]] == data_in.y[edge_index_added[1]]) * mask_labeled_edges).sum() / mask_labeled_edges.sum()
    ori_homo_ratio = ori_homo_ratio.cpu().item()
    print('Original added edge num:', edge_index_added.shape[1])
    print('Original homophily ratio:', ori_homo_ratio)
    pred_clf = probs_clf.argmax(dim=1)
    mask_remove_edge = torch.zeros(edge_index_added.shape[1]).bool()
    e_sim = e_sim.view(-1)
    num_edge_remove = 0
    # 1. remove low SimNet Confidence edges
    thres_conf = e_sim.quantile(q=thres_conf_quantile)
    mask_remove_edge[e_sim < thres_conf] = True
    num_edge_remove = mask_remove_edge.sum().item()
    print('1. remove low SimNet Confidence (<{:.3f}, quantile-{}) edges:'.format(thres_conf, thres_conf_quantile), num_edge_remove)
    # 2. remove edges that include node with wrong pred label compared with training label (ground truth)
    mask_remove_edge[(pred_clf[edge_index_added[0]] != data_in.y[edge_index_added[0]]) * data_in.train_mask[edge_index_added[1]]] = True
    mask_remove_edge[(pred_clf[edge_index_added[1]] != data_in.y[edge_index_added[1]]) * data_in.train_mask[edge_index_added[1]]] = True
    print('2. emove edges that include node with wrong pred label compared with training label (ground truth):', mask_remove_edge.sum() - num_edge_remove)
    num_edge_remove = mask_remove_edge.sum().item()
    # 3. remove edges that the clf predicted label of two nodes (end points) are different
    mask_remove_edge[pred_clf[edge_index_added[0]] != pred_clf[edge_index_added[1]]] = True
    print('3. remove edges that the clf predicted label of two nodes (end points) are different:', mask_remove_edge.sum() - num_edge_remove)
    num_edge_remove = mask_remove_edge.sum().item()
    # 4. remove low raw_feat_sim edges
    edge_cos_sim = F.cosine_similarity(data_in.x[edge_index_added[0]], data_in.x[edge_index_added[1]])
    mask_remove_edge[edge_cos_sim < thres_feat_sim] = True
    print('4. remove low raw_feat_sim edges:', mask_remove_edge.sum() - num_edge_remove)
    num_edge_remove = mask_remove_edge.sum().item()
    # Appliy mask_remove_edge
    new_edge_index_added = edge_index_added[:, ~mask_remove_edge]
    print('[Done] Totally remove edges: {} | Current total edge num: {}'.format(num_edge_remove, new_edge_index_added.shape[1]))
    # homophiy ratio of new cross-domain edges
    mask_labeled_edges = (data_in.y[new_edge_index_added[0]] != -1) * (data_in.y[new_edge_index_added[1]] != -1)
    new_homo_ratio = ((data_in.y[new_edge_index_added[0]] == data_in.y[new_edge_index_added[1]]) * mask_labeled_edges).sum() / mask_labeled_edges.sum()
    new_homo_ratio = new_homo_ratio.cpu().item()
    print('Current homophily ratio:', new_homo_ratio)
    return new_edge_index_added

def merge_graphs(data_src, data_tar, edge_index_cross_added, edge_index_added_src=None, edge_index_added_tar=None):
    N_src = data_src.x.shape[0]
    N_tar = data_tar.x.shape[0]
    N = N_src + N_tar
    x = torch.cat((data_src.x, data_tar.x), dim=0).cpu()
    edge_index_src_ori = data_src.edge_index.cpu()
    edge_index_tar_ori = (data_tar.edge_index + N_src).cpu()
    edge_index_cross_added[1, :] += N_src
    edge_index = torch.cat((edge_index_src_ori, edge_index_tar_ori, edge_index_cross_added), dim=1)
    if edge_index_added_src is not None:
        edge_index = torch.cat((edge_index, edge_index_added_src), dim=1)
    if edge_index_added_tar is not None:
        edge_index_added_tar = (edge_index_added_tar + N_src).cpu()
        edge_index = torch.cat((edge_index, edge_index_added_tar), dim=1)
    # edge_index = edge_index_cross_added

    # edge_index = torch.cat((edge_index_src_ori, edge_index_tar_ori), dim=1)

    central_mask = torch.zeros(N).bool()
    central_mask[:N_src] = True
    train_mask = torch.zeros(N).bool()
    val_mask = torch.zeros(N).bool()
    test_mask = torch.zeros(N).bool()
    train_mask[central_mask] = True
    train_mask[torch.where(data_src.y == -1)] = False
    train_mask[torch.where(data_tar.train_mask)[0] + N_src] = True
    val_mask[torch.where(data_tar.val_mask)[0] + N_src] = True
    test_mask[torch.where(data_tar.test_mask)[0] + N_src] = True
    y = torch.cat((data_src.y, data_tar.y), dim=0)
    return torch_geometric.data.Data(x=x, edge_index=edge_index, y=y, \
                                     train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, central_mask=central_mask).coalesce()

def reorder(data_merge, data_src, mapper_idx_src, mapper_idx_tar):
    # reorder for data_merge (keep consistent with order/node_id used in the original data)
    N_src = data_src.num_nodes
    mapper_merge = {}
    for key in mapper_idx_src:
        mapper_merge[key] = mapper_idx_src[key]
    for key in mapper_idx_tar:
        assert key not in mapper_merge
        mapper_merge[key] = mapper_idx_tar[key] + N_src
    mapper_merge_inverse = {}
    for key in mapper_merge:
        mapper_merge_inverse[mapper_merge[key]] = key
    mapper_merge_items_sort = sorted(list(mapper_merge.items()), reverse=False, key=lambda x:x[0])
    mapper_merge_items_sort = torch.LongTensor(mapper_merge_items_sort)
    reorder_idxs = mapper_merge_items_sort[:, 1]

    data_merge.train_mask = data_merge.train_mask[reorder_idxs]
    data_merge.val_mask = data_merge.val_mask[reorder_idxs]
    data_merge.test_mask = data_merge.test_mask[reorder_idxs]
    data_merge.central_mask = data_merge.central_mask[reorder_idxs]
    data_merge.x = data_merge.x[reorder_idxs]
    data_merge.y = data_merge.y[reorder_idxs]
    edge_index_ori = data_merge.edge_index
    # for i in range(2):
    #     for j in range(edge_index_ori.shape[1]):
    #         data_merge.edge_index[i, j] = mapper_merge_inverse[data_merge.edge_index[i, j].item()]
    data_merge.edge_index = torch.LongTensor([[mapper_merge_inverse[idx.item()] for idx in edge_index_ori[0]], [mapper_merge_inverse[idx.item()] for idx in edge_index_ori[1]]])
    return data_merge


def check_added_edges_cross_domain_validity(edge_index_added, e_sim, data_src, data_tar, probs_clf_src, probs_clf_tar, thres_conf_quantile=0.1, thres_feat_sim=0.):
    # homophiy ratio of orignal cross-domain edges
    mask_labeled_edges = (data_src.y[edge_index_added[0]] != -1) * (data_tar.y[edge_index_added[1]] != -1)
    ori_homo_ratio = ((data_src.y[edge_index_added[0]] == data_tar.y[edge_index_added[1]]) * mask_labeled_edges).sum() / mask_labeled_edges.sum()
    ori_homo_ratio = ori_homo_ratio.cpu().item()
    print('Original added edge num:', edge_index_added.shape[1])
    print('Original homophily ratio:', ori_homo_ratio)
    pred_clf_src = probs_clf_src.argmax(dim=1)
    pred_clf_tar = probs_clf_tar.argmax(dim=1)
    mask_remove_edge = torch.zeros(edge_index_added.shape[1]).bool()
    e_sim = e_sim.view(-1)
    num_edge_remove = 0
    # 1. remove low SimNet Confidence edges
    thres_conf = e_sim.quantile(q=thres_conf_quantile)
    mask_remove_edge[e_sim < thres_conf] = True
    num_edge_remove = mask_remove_edge.sum().item()
    print('1. remove low SimNet Confidence (<{:.3f}, quantile-{}) edges:'.format(thres_conf, thres_conf_quantile), num_edge_remove)
    # 2. remove edges that include node with wrong pred label compared with training label (ground truth)
    mask_remove_edge[pred_clf_src[edge_index_added[0]] != data_src.y[edge_index_added[0]]] = True
    mask_remove_edge[(pred_clf_tar[edge_index_added[1]] != data_tar.y[edge_index_added[1]]) * data_tar.train_mask[edge_index_added[1]]] = True
    print('2. emove edges that include node with wrong pred label compared with training label (ground truth):', mask_remove_edge.sum() - num_edge_remove)
    num_edge_remove = mask_remove_edge.sum().item()
    # 3. remove edges that the clf predicted label of two nodes (end points) are different
    mask_remove_edge[pred_clf_src[edge_index_added[0]] != pred_clf_tar[edge_index_added[1]]] = True
    print('3. remove edges that the clf predicted label of two nodes (end points) are different:', mask_remove_edge.sum() - num_edge_remove)
    num_edge_remove = mask_remove_edge.sum().item()
    # 4. remove low raw_feat_sim edges
    edge_cos_sim = F.cosine_similarity(data_src.x[edge_index_added[0]], data_tar.x[edge_index_added[1]])
    mask_remove_edge[edge_cos_sim < thres_feat_sim] = True
    print('4. remove low raw_feat_sim edges:', mask_remove_edge.sum() - num_edge_remove)
    num_edge_remove = mask_remove_edge.sum().item()
    # Appliy mask_remove_edge
    new_edge_index_added = edge_index_added[:, ~mask_remove_edge]
    print('[Done] Totally remove edges: {} | Current total edge num: {}'.format(num_edge_remove, new_edge_index_added.shape[1]))
    # homophiy ratio of new cross-domain edges
    mask_labeled_edges = (data_src.y[new_edge_index_added[0]] != -1) * (data_tar.y[new_edge_index_added[1]] != -1)
    new_homo_ratio = ((data_src.y[new_edge_index_added[0]] == data_tar.y[new_edge_index_added[1]]) * mask_labeled_edges).sum() / mask_labeled_edges.sum()
    new_homo_ratio = new_homo_ratio.cpu().item()
    print('Current homophily ratio:', new_homo_ratio)
    return new_edge_index_added


def gen_bridged_graph(args, data_src, data_tar, device, path_ckpt, mapper_idx_src, mapper_idx_tar, epsilon=0.5, batch_size=1000):
    if args.version == 'v1':
        sim_model = Adversarial_Learner(data_src, data_tar, dim_hidden=args.hidden_dim, num_layer=args.num_layer, source_clf=True, norm_mode=args.norm_mode, norm_scale=args.norm_scale)
    elif args.version == 'v2':
        # sim_model = Adversarial_Learner_v2(data_src, data_tar, dim_hidden=args.hidden_dim, num_layer=args.num_layer, source_clf=True, norm_mode=args.norm_mode, norm_scale=args.norm_scale)
        sim_model = Adversarial_Learner_v2(data_src, data_tar, dim_hidden=args.hidden_dim, num_layer=args.num_layer, use_norm=True, \
                                    source_clf=True, norm_mode=args.norm_mode, norm_scale=args.norm_scale, sim_mode=args.sim_mode, backbone=args.backbone)
    sim_model.load_state_dict(torch.load(path_ckpt))
    data_src = data_src.to(device)
    data_tar = data_tar.to(device)
    sim_model = sim_model.to(device)
    data_src = data_src.to(device)
    data_tar = data_tar.to(device)


    # add cross-domain edges
    # amazon2dslr: k=20
    # amazon2webcam: k=8
    edge_index_cross_domain_added, e_sim_mat, idx_src_mat, probs_clf_src, probs_clf_tar = \
        add_topk_sim_cross_domain_edges(data_src, data_tar, sim_model, epsilon=epsilon, k=args.k_cross, batch_size=batch_size)
    print(edge_index_cross_domain_added.shape, e_sim_mat.shape, idx_src_mat.shape, probs_clf_src.shape, probs_clf_tar.shape)
    if args.check_cross:
        edge_index_cross_domain_added = check_added_edges_cross_domain_validity(edge_index_cross_domain_added, e_sim_mat.view(-1), \
                                                           data_src, data_tar, probs_clf_src, probs_clf_tar, thres_conf_quantile=args.thres_conf_quantile, thres_feat_sim=args.thres_feat_sim)
    # print(e_sim_mat)

    if args.k_within > 0:
        # add within-domain edges
        edge_index_added_src, e_sim_mat_src, idx_src_mat_src = add_topk_sim_within_domain_edges(data_src, sim_model, k=args.k_within, batch_size=100, domain='source')
        edge_index_added_tar, e_sim_mat_tar, idx_src_mat_tar = add_topk_sim_within_domain_edges(data_tar, sim_model, k=args.k_within, batch_size=100, domain='target')
        print(edge_index_added_src.shape, edge_index_added_tar.shape)

        # validate the effectiveness of added edges
        # thres_feat_sim=0.9 for office, 0 for twitter
        if args.check_within:
            edge_index_added_src = check_added_edges_within_domain_validity(edge_index_added_src, e_sim_mat_src.view(-1), \
                                                                    data_src, probs_clf_src, thres_conf_quantile=0.1, thres_feat_sim=0.8)
            print('#' * 100)
            edge_index_added_tar = check_added_edges_within_domain_validity(edge_index_added_tar, e_sim_mat_tar.view(-1), \
                                                                    data_tar, probs_clf_tar, thres_conf_quantile=0.1, thres_feat_sim=0.8)
            print(edge_index_added_src.shape, edge_index_added_tar.shape)
    else:
        edge_index_added_src = edge_index_added_tar = None

    # merge graphs, adding valid edges
    data_merge = merge_graphs(data_src, data_tar, copy.deepcopy(edge_index_cross_domain_added), \
            copy.deepcopy(edge_index_added_src), copy.deepcopy(edge_index_added_tar))
    data_merge = reorder(data_merge, data_src, mapper_idx_src, mapper_idx_tar)
    eval_homophily(data_merge)
    eval_bridged_Graph(data_merge)
    if args.save:
        if not os.path.exists('../data_bridged_graph'):
            os.makedirs('../data_bridged_graph')
        torch.save(data_merge, f'../data_bridged_graph/{args.dataset_name}_bridged_graph.dat'), data_merge
    return data_merge



def main(args):
    set_random_seed(0)
    data_src, data_tar, data, mapper_idx_src, mapper_idx_tar = prepare_datasets(args.dataset_name)
    
    print(data)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    device
    print('Device:', device)

    if args.dataset_name.split('_')[0] == 'twitter':
        
        ori_edge_index_src = copy.deepcopy(data_src.edge_index)
        self_loop_src = torch.stack([torch.arange(data_src.num_nodes) for _ in range(2)], dim=0)
        data_src.edge_index = self_loop_src
        data_src, data_tar, ori_edge_index_src.shape
    
    print(data_src, data_tar)


    use_clf = True
    if args.version == 'v1':
        main_adv(args, data_src, data_tar, save=args.save, repeat=args.repeat, num_epoch=args.num_epoch, seed=args.seed, num_layer=args.num_layer, \
                        hidden= args.hidden_dim, metric='f1', use_clf=use_clf, norm_mode=args.norm_mode, norm_scale=args.norm_scale, \
                        eval_per_epoch=args.eval_per_epoch, start_eval_epoch=args.start_eval_epoch, device=device)
    elif args.version == 'v2':
        main_adv_v2(args, data_src, data_tar, save=args.save, repeat=args.repeat, num_epoch=args.num_epoch, seed=args.seed, num_layer=args.num_layer, \
                 hidden=args.hidden_dim, metric='f1', use_clf=use_clf, norm_mode=args.norm_mode, norm_scale=args.norm_scale, \
                 eval_per_epoch=args.eval_per_epoch, start_eval_epoch=args.start_eval_epoch, max_class_num=args.max_class_num, \
                 sample_size=args.sample_size, sim_mode=args.sim_mode, backbone=args.backbone, use_norm=True, eval_mode=args.eval_mode, device=device)

    gen_bridged_graph(args, data_src, data_tar, device, path_ckpt=f'../ckpt/model_AdvLearner_{args.dataset_name}_best.ckpt', \
                      mapper_idx_src=mapper_idx_src, mapper_idx_tar=mapper_idx_tar, epsilon=args.epsilon, batch_size=args.batch_size)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Training Scripts for Similarity Learner part of Bridged-GNN')
    ap.add_argument('--gpu', type=int, default=0, help='GPU ID')
    ap.add_argument('--dataset_name', type=str, default='twitter_unrelational')
    ap.add_argument('--save', action='store_true', default=False, help='save the model parameters')
    ap.add_argument('--check_within', action='store_true', default=False, help='whether to check the validity of added within-domain edges')
    ap.add_argument('--check_cross', action='store_true', default=False, help='whether to check the validity of added cross-domain edges')
    ap.add_argument('--norm_mode', type=str, default='None')
    ap.add_argument('--version', type=str, default='v1', choices=['v1', 'v2'], help='version of simlearner')
    ap.add_argument('--norm_scale', type=float, default=1.)
    ap.add_argument('--num_epoch', type=int, default=400)
    ap.add_argument('--start_eval_epoch', type=int, default=300)
    ap.add_argument('--eval_per_epoch', type=int, default=1)
    ap.add_argument('--num_layer', type=int, default=2)
    ap.add_argument('--hidden_dim', type=int, default=64)
    ap.add_argument('--sim_mode', type=str, default='mlp', choices=['cosine', 'mlp'])
    ap.add_argument('--backbone', type=str, default='mlp', choices=['gnn', 'mlp'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epsilon', type=float, default=0.5)
    ap.add_argument('--thres_conf_quantile', type=float, default=0.1)
    ap.add_argument('--thres_feat_sim', type=float, default=0.8)
    ap.add_argument('--k_within', type=int, default=6)
    ap.add_argument('--k_cross', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=1000)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--max_class_num', type=int, default=10)
    ap.add_argument('--eval_mode', type=str, default='sampling', choices=['all', 'sampling'])
    ap.add_argument('--sample_size', type=int, default=40000)

    

    args  = ap.parse_args()
    print(args)
    main(args=args)