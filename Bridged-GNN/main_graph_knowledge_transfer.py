import torch
from torch_geometric.nn import GATConv
import sys, os
sys.path.append('./models')
from models import *
from backbones import *
from KTGNN import KTGNN_no_complement, KTGNN_noDTC
import random
from utils import set_random_seed, eval_bridged_Graph, eval_homophily
import argparse
import numpy as np
import torch.nn.functional as F
from torch_geometric.transforms import ToUndirected
import torch_geometric, torch_sparse

class pyg_dataset:
    def __init__(self, data):
        self.num_classes = data.y.max().item() + 1
        self.num_nodes = data.x.shape[0]
        self.num_edges = data.edge_index.shape[1]
        self.num_features = data.x.shape[1]
        self.data = (data,)
    def __getitem__(self, idx):
        return self.data[idx]

# import functions for model training
import random
import time
import sys
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import networkx as nx
from torch.optim.lr_scheduler import StepLR
sys.path.append('./models')


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def train(data, model, optimizer, clip_grad=False, gnn=None, Lambda=1.):
    model.train()
    optimizer.zero_grad()
    if gnn == 'KTGNN':
        log_probs_xs, log_probs_xt, log_probs_xt_hat, loss_dist = model(data)
        loss_clf_s = F.nll_loss(log_probs_xs[data.train_mask], data.y[data.train_mask])
        loss_clf_t1 = F.nll_loss(log_probs_xt[data.train_mask * ~data.central_mask], data.y[data.train_mask * ~data.central_mask])
        loss_clf_t2 = F.nll_loss(log_probs_xt_hat[data.train_mask * ~data.central_mask], data.y[data.train_mask * ~data.central_mask])
        # loss_kl = F.kl_div(log_probs_xt_hat, log_probs_xt, log_target=True, reduction='batchmean') * 0.5 + F.kl_div(log_probs_xt, log_probs_xt_hat, log_target=True, reduction='batchmean') * 0.5
        loss_kl = F.kl_div(log_probs_xt_hat, log_probs_xt, log_target=True, reduction='batchmean')

        # loss_kl1 = F.kl_div(log_probs_xt_hat[~data.central_mask], log_probs_xt[~data.central_mask], log_target=True, reduction='batchmean')
        # loss_kl2 = F.kl_div(log_probs_xt_hat[~data.central_mask], log_probs_xs[~data.central_mask], log_target=True, reduction='batchmean')
        # loss_kl = loss_kl1 + loss_kl2
        # loss_kl = F.kl_div(log_probs_xt_hat, log_probs_xt, log_target=True, reduction='batchmean') + F.kl_div(log_probs_xt_hat, log_probs_xs, log_target=True, reduction='batchmean')
        loss = (loss_clf_s * 2. + loss_clf_t1 + loss_clf_t2) / 4. + loss_kl * Lambda
        print(loss_clf_s.cpu().item() , loss_clf_t1.cpu().item() , loss_clf_t2.cpu().item())
    else: 
        log_probs, loss_dist = model(data), None
        print(type(log_probs), type(data.train_mask), type(data.y))
        loss = F.nll_loss(log_probs[data.train_mask], data.y[data.train_mask])
    if loss_dist is not None:
        print('Loss_clf:{:.3f} | Loss_dist:{:.3f} | Loss_kl:{:.3f}'.format(loss.cpu().item(), loss_dist.cpu().item(), loss_kl.cpu().item()))
        # print('Loss_clf:{:.3f} | Loss_dist:{:.3f} | Loss_kl:{:.3f},{:.3f}'.format(loss.cpu().item(), loss_dist.cpu().item(), loss_kl1.cpu().item(), loss_kl2.cpu().item()))
        loss = loss + loss_dist
    else:
        print('Loss_clf:{:.3f} | Loss_kl:{:.3f}'.format(loss.cpu().item(), loss_kl.cpu().item()))
    loss.backward()
    optimizer.step()
    return loss.detach().item(), loss_clf_t2.detach().item(), loss_clf_t1.detach().item(), loss_kl.detach().item()


# @torch.no_grad()
# @torch.no_grad()
def test(data, model, dataset_name, gnn=None, metric='f1', f1_average='macro'):
    with torch.no_grad():
        model.eval()
        if gnn == 'KTGNN':
            log_probs_xs, log_probs_xt, log_probs_xt_hat, loss_dist = model(data)
            accs = []
            aucs = []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                if len(accs) == 0:
                    pred = log_probs_xs[mask].max(1)[1]
                    # score = pred.eq(data.y[mask]).sum().item() / mask.sum().item() # acc
                    if metric == 'f1':
                        num_classes = data.y.max().item() + 1
                        score = f1_score(data.y[mask].cpu().numpy(), pred.detach().cpu().numpy(), average=f1_average)
                    elif metric == 'auc':
                        score = roc_auc_score(data.y[mask].cpu().numpy(), log_probs_xs[mask, 1].detach().cpu().exp().numpy())
                    elif metric == 'acc':
                        score = accuracy_score(data.y[mask].cpu().numpy(), pred.detach().cpu().numpy())
                    else:
                        raise NotImplementedError('NotImplemented Metric:{}'.format(metric))
                else:
                    log_probs_eval = log_probs_xt_hat
                    # log_probs_eval = log_probs_xt.exp() + log_probs_xt_hat.exp()
                    pred = log_probs_eval[mask].max(1)[1]
                    # score = pred.eq(data.y[mask]).sum().item() / mask.sum().item() # acc
                    if metric == 'f1':
                        # num_classes = data.y.max().item() + 1
                        score = f1_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred.detach().cpu().numpy(), average=f1_average)
                    elif metric == 'auc':
                        score = roc_auc_score(data.y[mask * ~data.central_mask].cpu().numpy(), log_probs_eval[mask, 1].detach().cpu().exp().numpy())
                    elif metric == 'acc':
                        # num_classes = data.y.max().item() + 1
                        score = accuracy_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred.detach().cpu().numpy())
                    else:
                        raise NotImplementedError('NotImplemented Metric:{}'.format(metric))
                accs.append(score)
        else:
            log_probs = model(data)
            accs = []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = log_probs[mask].max(1)[1]
                # score = pred.eq(data.y[mask]).sum().item() / mask.sum().item() # acc
                num_classes = data.y.max().item() + 1
                score = f1_score(data.y[mask].cpu().numpy(), pred.detach().cpu().numpy(), average=f1_average)
                accs.append(score)
        return accs
def get_each_clf_res(data, model, metric='f1', f1_average='macro'):
    with torch.no_grad():
        model.eval()
        log_probs_xs, log_probs_xt, log_probs_xt_hat, loss_dist = model(data)
        accs = []
        mask = data.test_mask
        pred_src = log_probs_xs[mask].max(1)[1]
        pred_tar_hat = log_probs_xt_hat[mask].max(1)[1]
        pred_tar = log_probs_xt[mask].max(1)[1]
        if metric == 'f1':
            s1 = f1_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred_src.detach().cpu().numpy(), average=f1_average)
            s2 = f1_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred_tar.detach().cpu().numpy(), average=f1_average)
            s3 = f1_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred_tar_hat.detach().cpu().numpy(), average=f1_average)
        elif metric == 'auc':
            s1 = roc_auc_score(data.y[mask * ~data.central_mask].cpu().numpy(), log_probs_xs[mask, 1].detach().cpu().exp().numpy())
            s2 = roc_auc_score(data.y[mask * ~data.central_mask].cpu().numpy(), log_probs_xt[mask, 1].detach().cpu().exp().numpy())
            s3 = roc_auc_score(data.y[mask * ~data.central_mask].cpu().numpy(), log_probs_xt_hat[mask, 1].detach().cpu().exp().numpy())
        elif metric == 'acc':
            s1 = accuracy_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred_src.detach().cpu().exp().numpy())
            s2 = accuracy_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred_tar.detach().cpu().exp().numpy())
            s3 = accuracy_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred_tar_hat.detach().cpu().exp().numpy())
        else:
            raise NotImplementedError('NotImplemented Metric:{}'.format(metric))
        return [s1, s2, s3]
def train_gnn(args, dataset, data, save=False, repeat=3, num_epoch=200, gnn='GCN', seed=None, step_size=100, gamma=0.1,\
    num_layer=2, hidden= 64, lr=1e-3, wd=5e-3, use_shceduler=True, step=1, Lambda=1., f1_average='macro', metric='f1', noDTC=False):
    data = data.to(device)
    final_acc = {
            'train': [],
            'val': [],
            'test': []
        }
    loss_bucket = {
            'source&target': [],
            'target_hat': [],
            'target': [],
            'kl': []
        }
    for train_id in range(1, 1+repeat):
        print('repeat {}/{}'.format(train_id, repeat))
        
        data_split_seed = 0
        model_init_seed = train_id - 1
        if seed is not None:
            model_init_seed = seed

        set_random_seed(model_init_seed)

        
        if gnn == 'MLP':
            model = MLP(data.x.shape[1], dataset.num_classes, hidden=hidden)
        elif gnn == 'GCN':
            model = GCNNet(dataset, num_layer, hidden=hidden)
        elif gnn == 'GraphSAGE':
            model = GraphSAGE(dataset, num_layer, hidden, root_weight=True)
        elif gnn == 'GAT':
            model = GAT(dataset, hidden, head=3)
        elif gnn == 'GATv2':
            model = GATv2(dataset.num_features, hidden, dataset.num_classes, num_layer, heads=1, dropout=0.6, att_dropout=0.5)
        elif gnn == 'KTGNN':
            model = KTGNN_no_complement(dataset.num_features, data.y.max().item()+1, num_layer, hidden, root_weight=False, use_dist_loss=False, dropout=0.5, use_bn=True, step=step, dim_share=data.x.shape[1], need_complement=False)
        else:
            raise NotImplementedError('Not Implemented Model:{}'.format(gnn))
        # model = myGATv2(in_channels=dataset.num_features, hidden_channels=hidden, out_channels=dataset.num_classes, num_layers=num_layer, heads=1, dropout=0.5, att_dropout=0.5)
        # 
        
        model = model.to(device)
        print(data)
        print(model)

        # data_split_seed = int((train_id - 1) / 3)
        # model_init_seed = int((train_id - 1) % 3)
        # set_random_seed(data_split_seed)
        print('auto fixed data split seed to {}, model init seed to {}'.format(data_split_seed, model_init_seed))

        print(data)
        print('[Dataset-{}] train_num:{}, val_num:{}, test_num:{}, class_num:{}'.format(args.dataset_name, data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item(),data.y.max().item()+1))

        # # build optimizer
        # if gnn == 'KTGNN':
        #     optimizer = torch.optim.Adam([
        #                 {'params':model.param1,'weight_decay':wd},
        #                 {'params':model.param2,'weight_decay':1e-2},
        #                 {'params':model.param3,'weight_decay':1e-1},
        #                 ], lr=lr) # 5e-3 for pyg model, 5e-4 for idgl-implemented GCN
        # else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) # 5e-3 for pyg model, 5e-4 for idgl-implemented GCN
        if use_shceduler:
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=True)
        best_acc = {
            'train': 0,
            'val': 0,
            'test': 0,
            'loss': 666,
        }
        res_bucket_each = {
            'source&target': [],
            'target': [],
            'target_hat': []
        }
        for epoch in range(1, 1+num_epoch):
            t0 = time.time()
            loss_train, loss_target, loss_target_only, loss_kl = train(data, model, optimizer, gnn=gnn, Lambda=Lambda)
            loss_bucket['source&target'].append(loss_train)
            loss_bucket['target_hat'].append(loss_target)
            loss_bucket['target'].append(loss_target_only)
            loss_bucket['kl'].append(loss_kl)
            eval_res = test(data, model, dataset_name=args.dataset_name, gnn=gnn, metric=metric, f1_average=f1_average)
            eval_res_each = get_each_clf_res(data, model, metric='f1')
            res_bucket_each['source&target'].append(eval_res_each[0])
            res_bucket_each['target'].append(eval_res_each[1])
            res_bucket_each['target_hat'].append(eval_res_each[2])
            if use_shceduler:
                scheduler.step()
            log = 'Epoch: {:03d}, Loss:{:.4f} Train: {:.4f}, Val:{:.4f}, Test: {:.4f}, Time(s/epoch):{:.4f}'.format(epoch, loss_train, *eval_res, time.time() - t0)
            print(log)
            # if eval_res[1] > best_acc['val'] and loss_train < best_acc['loss']:
            # if eval_res[1] > best_acc['val']:
            # if loss_train < best_acc['loss']:
            if loss_target < best_acc['loss']:
                best_acc['train'] = eval_res[0]
                best_acc['val'] = eval_res[1]
                best_acc['test'] = eval_res[2]
                # best_acc['loss'] = loss_train
                best_acc['loss'] = loss_target
                if save:
                    torch.save(model.state_dict(), f'../ckpt/model_{gnn}_{args.dataset_name}_best.ckpt')
            
        print('[Run-{} score] {}'.format(train_id, best_acc))
        final_acc['train'].append(best_acc['train'])
        final_acc['val'].append(best_acc['val'])
        final_acc['test'].append(best_acc['test'])
    best_test_run  = np.argmax(final_acc['test'])
    final_acc_avg = {}
    final_acc_std = {}
    for key in final_acc:
        best_acc[key] = max(final_acc[key])
        final_acc_avg[key] = np.mean(final_acc[key])
        final_acc_std[key] = np.std(final_acc[key])
    print('[Average Score] {} '.format(final_acc_avg))
    print('[std Score] {} '.format(final_acc_std))
    print('[Best Score] {}'.format(best_acc))
    print('[Best test run] {}'.format(best_test_run))
    return loss_bucket, res_bucket_each


def train_noDTC(data, model, optimizer, gnn=None):
    model.train()
    optimizer.zero_grad()
    log_probs, loss_dist = model(data) if gnn == 'KTGNN' else (model(data), None)
    loss = F.nll_loss(log_probs[data.train_mask], data.y[data.train_mask])
    if loss_dist is not None:
        print('Loss_clf:{:.3f} | Loss_dist:{:.3f}'.format(loss.cpu().item(), loss_dist.cpu().item()))
        loss = loss + loss_dist
    loss.backward()
    optimizer.step()
    return loss.detach().item()


# @torch.no_grad()
def test_noDTC(data, model, gnn=None, metric='f1', f1_average='macro'):
    with torch.no_grad():
        model.eval()
        log_probs, _ = model(data) if gnn == 'KTGNN' else (model(data), None)
        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = log_probs[mask].max(1)[1]
            # score = pred.eq(data.y[mask]).sum().item() / mask.sum().item() # acc
            score = f1_score(data.y[mask].cpu().numpy(), pred.detach().cpu().numpy(), average='macro')
            if metric == 'f1':
                # num_classes = data.y.max().item() + 1
                score = f1_score(data.y[mask].cpu().numpy(), pred.detach().cpu().numpy(), average=f1_average)
            elif metric == 'auc':
                score = roc_auc_score(data.y[mask].cpu().numpy(), log_probs[mask, 1].detach().cpu().exp().numpy())
            elif metric == 'acc':
                # num_classes = data.y.max().item() + 1
                score = accuracy_score(data.y[mask].cpu().numpy(), pred.detach().cpu().numpy())
            else:
                raise NotImplementedError('NotImplemented Metric:{}'.format(metric))

            accs.append(score)
        return accs

def train_gnn_noDTC(args, dataset, data, save=False, repeat=3, num_epoch=200, gnn='GCN', seed=None, num_layer=2, hidden= 64, \
               lr=1e-3, wd=5e-3, use_scheduler=True, step=1, step_size=100, gamma=0.1, metric='f1', f1_average='macro'):
    data = data.to(device)
    final_acc = {
            'train': [],
            'val': [],
            'test': []
        }
    for train_id in range(1, 1+repeat):
        print('repeat {}/{}'.format(train_id, repeat))
        
        data_split_seed = 0
        model_init_seed = train_id - 1
        if seed is not None:
            model_init_seed = seed

        set_random_seed(model_init_seed)

        
        if gnn == 'MLP':
            model = MLP(data.x.shape[1], dataset.num_classes, hidden=hidden)
        elif gnn == 'GCN':
            model = GCNNet(dataset, num_layer, hidden=hidden)
        elif gnn == 'GraphSAGE':
            model = GraphSAGE(dataset, num_layer, hidden, root_weight=True)
        elif gnn == 'GAT':
            model = GAT(dataset, hidden, head=3)
        elif gnn == 'GATv2':
            model = GATv2(dataset.num_features, hidden, dataset.num_classes, num_layer, heads=1, dropout=0.6, att_dropout=0.5)
        elif gnn == 'KTGNN':
            model = KTGNN_noDTC(dataset.num_features, data.y.max().item()+1, num_layer, hidden, root_weight=False, use_dist_loss=False, \
                                     dropout=0.5, use_bn=True, step=step, dim_share=data.x.shape[1], need_complement=False)
            # model = myGATv2(dataset.num_features, hidden, dataset.num_classes, num_layer, heads=3, dropout=0.6, att_dropout=0.5)
        else:
            raise NotImplementedError('Not Implemented Model:{}'.format(gnn))
        # model = myGATv2(in_channels=dataset.num_features, hidden_channels=hidden, out_channels=dataset.num_classes, num_layers=num_layer, heads=1, dropout=0.5, att_dropout=0.5)
        # 
        
        model = model.to(device)
        print(data)
        print(model)

        # data_split_seed = int((train_id - 1) / 3)
        # model_init_seed = int((train_id - 1) % 3)
        # set_random_seed(data_split_seed)
        print('auto fixed data split seed to {}, model init seed to {}'.format(data_split_seed, model_init_seed))

        print(data)
        print('[Dataset-{}] train_num:{}, val_num:{}, test_num:{}, class_num:{}'.format(dataset, data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item(), dataset.num_classes))

        # build optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) # 5e-3 for pyg model, 5e-4 for idgl-implemented GCN
        if use_scheduler:
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=True)
        best_acc = {
            'train': 0,
            'val': 0,
            'test': 0,
            'loss': 666,
        }

        for epoch in range(1, 1+num_epoch):
            t0 = time.time()
                
            loss_train = train_noDTC(data, model, optimizer, gnn=gnn)
            eval_res = test_noDTC(data, model, gnn=gnn, metric=metric, f1_average=f1_average)
            if use_scheduler:
                scheduler.step()
            log = 'Epoch: {:03d}, Loss:{:.4f} Train: {:.4f}, Val:{:.4f}, Test: {:.4f}, Time(s/epoch):{:.4f}'.format(epoch, loss_train, *eval_res, time.time() - t0)
            print(log)
            # if eval_res[1] > best_acc['val'] and loss_train < best_acc['loss']:
            # if eval_res[1] > best_acc['val']:
            if loss_train < best_acc['loss']:
                best_acc['train'] = eval_res[0]
                best_acc['val'] = eval_res[1]
                best_acc['test'] = eval_res[2]
                best_acc['loss'] = loss_train
                if save:
                    torch.save(model.state_dict(), f'../ckpt/model_{gnn}_{args.dataset_name}_share_best.ckpt')
            
        print('[Run-{} score] {}'.format(train_id, best_acc))
        final_acc['train'].append(best_acc['train'])
        final_acc['val'].append(best_acc['val'])
        final_acc['test'].append(best_acc['test'])
    best_test_run  = np.argmax(final_acc['test'])
    final_acc_avg = {}
    final_acc_std = {}
    for key in final_acc:
        best_acc[key] = max(final_acc[key])
        final_acc_avg[key] = np.mean(final_acc[key])
        final_acc_std[key] = np.std(final_acc[key])
    print('[Average Score] {} '.format(final_acc_avg))
    print('[std Score] {} '.format(final_acc_std))
    print('[Best Score] {}'.format(best_acc))
    print('[Best test run] {}'.format(best_test_run))


def main(args):
    set_random_seed(0)
    data = torch.load(args.path_data)
    print(data)
    eval_bridged_Graph(data)
    data.train_mask[data.y==-1] = False
    dataset = pyg_dataset(data)

    # for twitter, use_DTC
    step_size = 100
    gamma = 0.1
    if args.to_undirected:
        ToUndirected(merge=True)(data) # optional for twitter (auc higher but f1 lower)

    
    if args.no_dtc:
        train_gnn_noDTC(args, dataset, data, save=False, repeat=1, num_epoch=args.num_epoch, gnn='GraphSAGE', \
                        seed=0, num_layer=args.num_layer, hidden=args.hidden_dim, lr=1e-3, wd=5e-3, \
                        use_scheduler=False, step=1, step_size=step_size, gamma=gamma, metric=args.eval_metric, f1_average='macro')
    else:
        loss_bucket, res_bucket_each = train_gnn(args, dataset, data, save=False, repeat=1, num_epoch=args.num_epoch,  step_size=step_size, gamma=gamma,\
                                gnn=args.model_name, seed=0, num_layer=args.num_layer, hidden=args.hidden_dim, lr=1e-3, \
                                wd=5e-3, use_shceduler=True, step=1, Lambda=1., metric=args.eval_metric, f1_average='macro')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Training Scripts for Similarity Learner part of Bridged-GNN')
    ap.add_argument('--gpu', type=int, default=0, help='GPU ID')
    ap.add_argument('--dataset_name', type=str, default='twitter_unrelational')
    ap.add_argument('--model_name', type=str, default='KTGNN', choices=['MLP', 'GCN', 'GAT', 'GATv2', 'GraphSAGE', 'KTGNN'])
    ap.add_argument('--eval_metric', type=str, default='f1', choices=['f1', 'auc'])
    ap.add_argument('--save', action='store_true', default=False, help='save the model parameters')
    ap.add_argument('--to_undirected', action='store_true', default=False)
    ap.add_argument('--no_dtc', action='store_true', default=False, help='use no_DTC version of KTGNN')
    ap.add_argument('--num_layer', type=int, default=2)
    ap.add_argument('--num_epoch', type=int, default=300)
    ap.add_argument('--hidden_dim', type=int, default=64)
    ap.add_argument('--path_data', type=str, default='/home/biwendong/code/knowledge_transferable_GNN/general_knowledge_transfer/code_cikm2023/data_bridged_graph/twitter_unrelational_bridged_graph.dat')
    
    args  = ap.parse_args()
    print(args)
    main(args=args)
