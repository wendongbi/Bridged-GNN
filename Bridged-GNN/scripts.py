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

def train_adv_few_shot(epoch, data_src, data_tar, model, optimizer_src_tar, optimizer_D, metric='f1', \
                pair_enumerator_src_train=None, pair_enumerator_tar_train=None, pair_enumerator_cross_train=None, max_class_num=2, sample_size=10000, use_clf=False):
    model.train()
    # 1. Train Similarity-Learner & Auto-Encoder
    # within-source domain similarity
    optimizer_src_tar.zero_grad()
    idx1_src, idx2_src = pair_enumerator_src_train.sampling(max_class_num=max_class_num, sample_size=sample_size, shuffle=False)
    probs_pair_src, log_probs_clf_src, h_src = model.source_learner(data_src, idx1_src, idx2_src, return_representation=True)
    y_pair_src = (data_src.y[idx1_src] == data_src.y[idx2_src]).float().unsqueeze(-1)
    loss_sim_within_src = F.binary_cross_entropy(probs_pair_src, y_pair_src)
    # within-target domain similarity
    idx1_tar, idx2_tar = pair_enumerator_tar_train.sampling(max_class_num=max_class_num, sample_size=sample_size, shuffle=False)
    # h_tar, _ = model.target_learner.encode(data_tar)
    h0_tar, h_tar, recons = model.target_learner(data_tar)
    probs_pair_tar, log_probs_clf_tar = model.source_learner.sim_net(h_tar, idx1_tar, idx2_tar)
    y_pair_tar = (data_tar.y[idx1_tar] == data_tar.y[idx2_tar]).float().unsqueeze(-1)
    loss_sim_within_tar = F.binary_cross_entropy(probs_pair_tar, y_pair_tar)
    # cross-domain similarity
    idx1_cross, idx2_cross = pair_enumerator_cross_train.sampling(max_class_num=max_class_num, sample_size=sample_size, shuffle=False)
    y_pair_cross = (data_src.y[idx1_cross] == data_tar.y[idx2_cross]).float().unsqueeze(-1)
    probs_pair_cross = model.source_learner.sim_net.similarity_cross_domain(h_src, h_tar, idx1_cross, idx2_cross).unsqueeze(-1)
    loss_sim_cross = F.binary_cross_entropy(probs_pair_cross, y_pair_cross)
    # AutoEncoder
    loss_recons = F.mse_loss(recons, h0_tar)
    # if epoch > 0:
    g_labels = torch.ones((h_tar.shape[0], 1)).to(h_tar.device)
    loss_g = F.binary_cross_entropy(model.discriminator(h_tar), g_labels)
    loss_ae = loss_g + loss_recons * 0.1
    # else:
    #     loss_ae = loss_recons

    loss_sim = loss_sim_within_src + loss_sim_within_tar + loss_sim_cross + loss_ae
    if use_clf:
        loss_clf_src = F.nll_loss(log_probs_clf_src[data_src.train_mask], data_src.y[data_src.train_mask])
        loss_clf_tar = F.nll_loss(log_probs_clf_tar[data_tar.train_mask], data_tar.y[data_tar.train_mask])
        loss_sim += loss_clf_src + loss_clf_tar
        print('Loss_sim:{:.4f} | Loss_clf_src:{:.4f} | Loss_clf_tar:{:.4f}'.format(loss_sim.detach().cpu().item(), loss_clf_src.detach().cpu().item(), loss_clf_tar.detach().cpu().item()))
    loss_sim.backward()
    optimizer_src_tar.step()
    pred_pair_src = (probs_pair_src > 0.5).long().view(-1)
    pred_pair_tar = (probs_pair_tar > 0.5).long().view(-1)
    pred_pair_cross = (probs_pair_cross > 0.5).long().view(-1)
    # print((y_pair == 1).sum().item(), (y_pair == 0).sum().item(), accuracy_score(y_pair.long().cpu().numpy(), pred_pair.detach().cpu().numpy()))
    if metric == 'f1':
        eval_src = f1_score(y_pair_src.long().cpu().numpy(), pred_pair_src.detach().cpu().numpy(), average='binary')
        eval_tar = f1_score(y_pair_tar.long().cpu().numpy(), pred_pair_tar.detach().cpu().numpy(), average='binary')
        eval_cross = f1_score(y_pair_cross.long().cpu().numpy(), pred_pair_cross.detach().cpu().numpy(), average='binary')
        eval_pair = (eval_src, eval_tar, eval_cross)
    elif metric == 'auc':
        eval_src = roc_auc_score(y_pair_src.long().cpu().numpy(), pred_pair_src.detach().cpu().numpy())
        eval_tar = f1_score(y_pair_tar.long().cpu().numpy(), pred_pair_tar.detach().cpu().numpy(), average='binary')
        eval_cross = f1_score(y_pair_cross.long().cpu().numpy(), pred_pair_cross.detach().cpu().numpy(), average='binary')
        eval_pair = (eval_src, eval_tar, eval_cross)
    else:
        raise NotImplementedError('NotImplemented Metric:{}'.format(metric))
        
    # 2. Train Discriminator
    optimizer_D.zero_grad()
    real_labels = torch.ones((h_src.shape[0], 1)).to(h_src.device)
    fake_labels = torch.zeros((h_tar.shape[0], 1)).to(h_tar.device)
    real_loss = F.binary_cross_entropy(model.discriminator(h_src.detach()), real_labels)
    fake_loss = F.binary_cross_entropy(model.discriminator(h_tar.detach()), fake_labels)
    loss_d = (real_loss + fake_loss) / 2
    loss_d.backward()
    optimizer_D.step()

    return loss_sim.detach().item(), eval_pair, loss_d.detach().item(), loss_ae.detach().item(), loss_g.detach().item(), loss_recons.detach().item()

# 测试：Cross-domain edge prediction
from sklearn.metrics import precision_score, recall_score
def eval_cross_domain(data_src, data_tar, model, mode='test', conf_lower_bound=None):
    mask_src = data_src.val_mask if mode=='val' else data_src.test_mask
    mask_tar = data_tar.train_mask + data_tar.val_mask if mode=='val' else data_tar.train_mask + data_tar.test_mask + data_tar.val_mask
    # mask_tar = data_tar.val_mask + data_tar.test_mask
    sample_idxs_1 = torch.where(mask_src)[0]
    sample_idxs_2 = torch.where(mask_tar)[0]
    pair_idxs = pair_enumeration(sample_idxs_1.unsqueeze(1), sample_idxs_2.unsqueeze(1)).transpose(0, 1) # (2, sample_size)
    idx1 = pair_idxs[0]
    idx2 = pair_idxs[1]

    mask_src = data_src.train_mask if mode == 'val' else data_src.train_mask + data_src.val_mask
    mask_tar = data_tar.val_mask if mode == 'val' else data_tar.test_mask
    sample_idxs_1 = torch.where(mask_src)[0]
    sample_idxs_2 = torch.where(mask_tar)[0]
    pair_idxs = pair_enumeration(sample_idxs_1.unsqueeze(1), sample_idxs_2.unsqueeze(1)).transpose(0, 1) # (2, sample_size)
    idx1 = torch.cat((idx1, pair_idxs[0]), dim=-1)
    idx2 = torch.cat((idx2, pair_idxs[1]), dim=-1)
    metric = 'f1'
    with torch.no_grad():
        model.eval()
        # target_enumerator = Pair_Enumerator(data_tar, mode='all')
        # idx1, idx2 = target_enumerator.sampling(max_class_num=2, sample_size=40000, shuffle=False)
        y_pair = (data_src.y[idx1] == data_tar.y[idx2]).float().unsqueeze(-1)
        probs_pair, log_probs_clf_src, log_probs_clf_tar, h_src, h_tar = model.get_probs_cross_domain(data_src, data_tar, idx1, idx2, return_representation=True)
        pred_pair = (probs_pair > 0.5).long().view(-1)
        # print('pred_pos_rate:', (pred_pair == 1).sum() / pred_pair.shape[0])
        # print('precision:{:.4f} | recall:{:.4f}'.format(precision_score(y_pair.long().cpu().numpy(), pred_pair.detach().cpu().numpy()),\
        #     recall_score(y_pair.long().cpu().numpy(), pred_pair.detach().cpu().numpy())))
        if conf_lower_bound is not None:
            assert conf_lower_bound <= 1 and conf_lower_bound >= 0
            high_conf = np.quantile(probs_pair.cpu().numpy(), q=conf_lower_bound)
            low_conf = np.quantile(probs_pair.cpu().numpy(), q=1-conf_lower_bound)
            print('low_conf:{:.4f} | high_conf:{:.4f}'.format(low_conf, high_conf))
            mask_conf = ((probs_pair >= high_conf) + (probs_pair <= low_conf)).view(-1).bool()
            print('Num of high-conf-samples:{}/{}'.format(mask_conf.sum().item(), mask_conf.shape[0]))
        else:
            mask_conf = torch.ones(probs_pair.shape[0]).bool()
        if metric == 'f1':
            score_pair = f1_score(y_pair[mask_conf].long().cpu().numpy(), pred_pair[mask_conf].detach().cpu().numpy(), average='binary')
        elif metric == 'acc':
            score_pair = accuracy_score(y_pair[mask_conf].long().cpu().numpy(), pred_pair[mask_conf].detach().cpu().numpy())
        elif metric == 'auc':
            score_pair = roc_auc_score(y_pair[mask_conf].long().cpu().numpy(), pred_pair[mask_conf].detach().cpu().numpy())
        else:
            raise NotImplementedError('NotImplemented Metric:{}'.format(metric))
    return score_pair


# 测试：Within-target domain
# 还需要测试Within-source domain和Cross-domain预测的准确度
def eval_within_domain(data, model, mode='test', domain='target', conf_lower_bound=None):
    mask_1 = data.train_mask + data.val_mask + data.test_mask
    mask_2 = data.val_mask if mode == 'val' else data.test_mask
    sample_idxs_1 = torch.where(mask_1)[0]
    sample_idxs_2 = torch.where(mask_2)[0]
    pair_idxs = pair_enumeration(sample_idxs_1.unsqueeze(1), sample_idxs_2.unsqueeze(1)).transpose(0, 1) # (2, sample_size)

    idx1 = pair_idxs[0]
    idx2 = pair_idxs[1]
    metric = 'f1'
    with torch.no_grad():
        model.eval()
        y_pair = (data.y[idx1] == data.y[idx2]).float().unsqueeze(-1)
        probs_pair, log_probs_clf = model.get_probs_within_domain(data, idx1, idx2, domain=domain)
        pred_pair = (probs_pair > 0.5).long().view(-1)
        pred_clf =  log_probs_clf[mask_2].max(1)[1]

        if conf_lower_bound is not None:
            assert conf_lower_bound <= 1 and conf_lower_bound >= 0
            high_conf = np.quantile(probs_pair.cpu().numpy(), q=conf_lower_bound)
            low_conf = np.quantile(probs_pair.cpu().numpy(), q=1-conf_lower_bound)
            print('low_conf:{:.4f} | high_conf:{:.4f}'.format(low_conf, high_conf))
            mask_conf = ((probs_pair >= high_conf) + (probs_pair <= low_conf)).view(-1).bool()
            print('Num of high-conf-samples:{}/{}'.format(mask_conf.sum().item(), mask_conf.shape[0]))
        else:
            mask_conf = torch.ones(probs_pair.shape[0]).bool()

        if metric == 'f1':
            score_pair = f1_score(y_pair[mask_conf].long().cpu().numpy(), pred_pair[mask_conf].detach().cpu().numpy(), average='binary')
            score_clf = f1_score(data.y[mask_2].long().cpu().numpy(), pred_clf.detach().cpu().numpy(), average='binary' if max(data.y) <= 1 else 'macro')
        elif metric == 'auc':
            score_pair = roc_auc_score(y_pair[mask_conf].long().cpu().numpy(), pred_pair[mask_conf].detach().cpu().numpy())
            score_clf = roc_auc_score(data.y[mask_2].long().cpu().numpy(), pred_clf.detach().cpu().numpy())
        else:
            raise NotImplementedError('NotImplemented Metric:{}'.format(metric))
    return score_pair, score_clf

def eval_adv(data_src, data_tar, model, mode='test'):
    eval_pair_src, eval_clf_src = eval_within_domain(data_src, model, mode=mode, domain='source')
    eval_pair_tar, eval_clf_tar = eval_within_domain(data_tar, model, mode=mode, domain='target')
    eval_pair_cross = eval_cross_domain(data_src, data_tar, model, mode=mode)
    return (eval_pair_src, eval_clf_src, eval_pair_tar, eval_clf_tar, eval_pair_cross)


def main_adv(args, data_src, data_tar, save=False, repeat=3, num_epoch=200, seed=None, num_layer=2, hidden= 64, metric='f1', \
             use_clf=True, norm_mode='PN', norm_scale=1., eval_per_epoch=1, start_eval_epoch=0, sim_mode='mlp', backbone='mlp', device=None):
    assert device is not None
    data_src = data_src.to(device)
    data_tar = data_tar.to(device)
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
        print('auto fixed data split seed to {}, model init seed to {}'.format(data_split_seed, model_init_seed))

        # model = Adversarial_Learner(data, dim_hidden=64, norm_mode='PN', norm_scale=1, use_clf=use_clf)
        if args.version == 'v1':
            model = Adversarial_Learner(data_src, data_tar, dim_hidden=hidden, num_layer=2, source_clf=use_clf, norm_mode=norm_mode, norm_scale=1.)
        elif args.version == 'v2':
            model = Adversarial_Learner_v2(data_src, data_tar, dim_hidden=hidden, num_layer=2, use_norm=True, \
                                    source_clf=use_clf, norm_mode=norm_mode, norm_scale=norm_scale, sim_mode=sim_mode, backbone=backbone)
        # model.set_device(device)
        model = model.to(device)
        print(model)
        

        print(data_src, data_tar)
        # build optimizer
        lr = 1e-3
        b1 = 0.5
        b2 = 0.999
        clip_value = 0.01
        optimizer_src_tar = torch.optim.Adam([
            {'params': model.source_learner.parameters(), 'lr': 1e-2, 'weight_decay': 5e-3},
            {'params': model.target_learner.parameters(), 'lr': lr, 'betas': (b1, b2)},
        ])
        # optimizer_AE = torch.optim.Adam(model.target_learner.parameters(), lr=lr, betas=(b1, b2)) 
        optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(b1, b2)) 

        best_acc = {
            'epoch': -1,
            'train': (0,0,0),
            'val': (0,0,0),
            'test': (0,0,0),
            'loss': 666,
        }
        # Train Source-Learner
        pair_enumerator_src_train = Pair_Enumerator(data_src, mode='train')
        pair_enumerator_tar_train = Pair_Enumerator(data_tar, mode='train')
        pair_enumeration_cross_train = Pair_Enumerator_cross(data_src, data_tar, mode='train')

        
        
        
        for epoch in range(1, 1+num_epoch):
            t0 = time.time()
            loss_sim, eval_pair_train, loss_d, loss_ae, loss_g, loss_recons = train_adv_few_shot(epoch, data_src, data_tar, model, optimizer_src_tar, optimizer_D, metric=metric, \
                    pair_enumerator_src_train=pair_enumerator_src_train, pair_enumerator_tar_train=pair_enumerator_tar_train, \
                    pair_enumerator_cross_train=pair_enumeration_cross_train, max_class_num=2, sample_size=40000, use_clf=use_clf)

            print(''.format(loss_g, loss_recons))
            log_ae = '[AE]Epoch: {:03d}, Loss_ae:{:.4f} | Loss_recons:{:.4f} | Loss_g:{:.4f} | Loss_d:{:.4f}  Time(s/epoch):{:.4f}'.format(
                epoch, loss_ae, loss_recons, loss_g, loss_d, time.time() - t0)
            print(log_ae)
            
            # val_pair, val_clf = eval_sim(data_src, model.source_learner, metric, pair_enumerator_val, data_src.val_mask, max_class_num=2, sample_size=20000, use_clf=use_clf)
            # test_pair, test_clf = eval_sim(data_src, model.source_learner, metric, pair_enumerator_test, data_src.test_mask, max_class_num=2, sample_size=20000, use_clf=use_clf)
            if epoch >= start_eval_epoch and epoch % eval_per_epoch == 0:
                eval_pair_src_val, eval_clf_src_val, eval_pair_tar_val, eval_clf_tar_val, eval_pair_cross_val = eval_adv(data_src, data_tar, model, mode='val')
                eval_pair_src_test, eval_clf_src_test, eval_pair_tar_test, eval_clf_tar_test, eval_pair_cross_test = eval_adv(data_src, data_tar, model, mode='test')

                if use_clf:
                    log_source = '[Sim]Epoch: {:03d}, Loss:{:.4f} | Train Pair:{:.4f}/{:.4f}/{:.4f} | Val Pair:{:.4f}/{:.4f}/{:.4f} | Test Pair:{:.4f}/{:.4f}/{:.4f} | Val CLF:{:.4f}/{:.4f} | Test CLF:{:.4f}/{:.4f} | Time(s/epoch):{:.4f}'.format(\
                        epoch, loss_sim, *eval_pair_train, *(eval_pair_src_val, eval_pair_tar_val, eval_pair_cross_val), *(eval_pair_src_test, eval_pair_tar_test, eval_pair_cross_test), \
                        eval_clf_src_val, eval_clf_tar_val, eval_clf_src_test, eval_clf_tar_test, time.time() - t0)
                else:
                    log_source = '[Sim]Epoch: {:03d}, Loss:{:.4f} | Train Pair:{:.4f}/{:.4f}/{:.4f} | Val Pair:{:.4f}/{:.4f}/{:.4f} | Test Pair:{:.4f}/{:.4f}/{:.4f}| Time(s/epoch):{:.4f}'.format(\
                        epoch, loss_sim, *eval_pair_train, *(eval_pair_src_val, eval_pair_tar_val, eval_pair_cross_val), *(eval_pair_src_test, eval_pair_tar_test, eval_pair_cross_test), time.time() - t0)
                print(log_source)
                # if  eval_pair_cross_val > best_acc['val'][2] and loss_sim < best_acc['loss']:
                if eval_pair_cross_val > best_acc['val'][2]:
                # if loss_sim < best_acc['loss']:
                    best_acc['train'] = eval_pair_train
                    best_acc['val'] = (eval_pair_src_val, eval_pair_tar_val, eval_pair_cross_val)
                    best_acc['test'] = (eval_pair_src_test, eval_pair_tar_test, eval_pair_cross_test)
                    best_acc['loss'] = loss_sim
                    best_acc['epoch'] = epoch
                    if save:
                        torch.save(model.state_dict(), f'../ckpt/model_AdvLearner_{args.dataset_name}_best.ckpt')
            # # Train Target-Learner
            # t0 = time.time()
            # loss_train = train_ae(data_tar, model.target_learner, optimizer_AE)
            # # eval_res = test(data, model, dataset_name=dataset_name, metric=metric)
            # log = 'Epoch: {:03d}, Loss:{:.4f}  Time(s/epoch):{:.4f}'.format(epoch, loss_train, time.time() - t0)
            # print(log)
        if save:
            torch.save(model.state_dict(), f'../ckpt/model_AdvLearner_{args.dataset_name}_final.ckpt')   
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




from sklearn.metrics import precision_score, recall_score
def eval_cross_domain_v2(data_src, data_tar, model, pair_enumerator=None, split='test', conf_lower_bound=None, metric = 'f1', eval_mode='sampling'):
    if eval_mode == 'all':
        mask_src = data_src.val_mask if split=='val' else data_src.test_mask
        mask_tar = data_tar.train_mask + data_tar.val_mask if split=='val' else data_tar.train_mask + data_tar.test_mask + data_tar.val_mask
        # mask_tar = data_tar.val_mask + data_tar.test_mask
        sample_idxs_1 = torch.where(mask_src)[0]
        sample_idxs_2 = torch.where(mask_tar)[0]
        pair_idxs = pair_enumeration(sample_idxs_1.unsqueeze(1), sample_idxs_2.unsqueeze(1)).transpose(0, 1) # (2, sample_size)
        idx1 = pair_idxs[0]
        idx2 = pair_idxs[1]

        mask_src = data_src.train_mask if split == 'val' else data_src.train_mask + data_src.val_mask
        mask_tar = data_tar.val_mask if split == 'val' else data_tar.test_mask
        sample_idxs_1 = torch.where(mask_src)[0]
        sample_idxs_2 = torch.where(mask_tar)[0]
        pair_idxs = pair_enumeration(sample_idxs_1.unsqueeze(1), sample_idxs_2.unsqueeze(1)).transpose(0, 1) # (2, sample_size)
        idx1 = torch.cat((idx1, pair_idxs[0]), dim=-1)
        idx2 = torch.cat((idx2, pair_idxs[1]), dim=-1)
    elif eval_mode == 'sampling':
        num_classes = data_tar.y.max().item() + 1
        idx1, idx2 = pair_enumerator.balanced_sampling(max_class_num=num_classes, sample_size=100000, shuffle=False)
    else:
        raise NotImplementedError('Not Implemented Eval Mode:{}'.format(eval_mode))
    
    with torch.no_grad():
        model.eval()
        # target_enumerator = Pair_Enumerator(data_tar, mode='all')
        # idx1, idx2 = target_enumerator.sampling(max_class_num=2, sample_size=40000, shuffle=False)
        y_pair = (data_src.y[idx1] == data_tar.y[idx2]).float().unsqueeze(-1)
        probs_pair, log_probs_clf_src, log_probs_clf_tar, h_src, h_tar = model.get_probs_cross_domain(data_src, data_tar, idx1, idx2, return_representation=True)
        pred_pair = (probs_pair > 0.5).long().view(-1)
        # print('#'*10, 'cross', pred_pair.sum())
        # print('pred_pos_rate:', (pred_pair == 1).sum() / pred_pair.shape[0])
        # print('precision:{:.4f} | recall:{:.4f}'.format(precision_score(y_pair.long().cpu().numpy(), pred_pair.detach().cpu().numpy()),\
        #     recall_score(y_pair.long().cpu().numpy(), pred_pair.detach().cpu().numpy())))
        if conf_lower_bound is not None:
            assert conf_lower_bound <= 1 and conf_lower_bound >= 0
            high_conf = np.quantile(probs_pair.cpu().numpy(), q=conf_lower_bound)
            low_conf = np.quantile(probs_pair.cpu().numpy(), q=1-conf_lower_bound)
            print('low_conf:{:.4f} | high_conf:{:.4f}'.format(low_conf, high_conf))
            mask_conf = ((probs_pair >= high_conf) + (probs_pair <= low_conf)).view(-1).bool()
            print('Num of high-conf-samples:{}/{}'.format(mask_conf.sum().item(), mask_conf.shape[0]))
        else:
            mask_conf = torch.ones(probs_pair.shape[0]).bool()
        if metric == 'f1':
            score_pair = f1_score(y_pair[mask_conf].long().cpu().numpy(), pred_pair[mask_conf].detach().cpu().numpy(), average='binary')
        elif metric == 'acc':
            score_pair = accuracy_score(y_pair[mask_conf].long().cpu().numpy(), pred_pair[mask_conf].detach().cpu().numpy())
        elif metric == 'auc':
            score_pair = roc_auc_score(y_pair[mask_conf].long().cpu().numpy(), pred_pair[mask_conf].detach().cpu().numpy())
        else:
            raise NotImplementedError('NotImplemented Metric:{}'.format(metric))
    return score_pair


# 测试：Within-target domain
# 还需要测试Within-source domain和Cross-domain预测的准确度
def eval_within_domain_v2(data, model, pair_enumerator=None, split='test', domain='target', conf_lower_bound=None, metric='f1', eval_mode='sampling'):
    if eval_mode == 'all':
        mask_1 = data.train_mask + data.val_mask + data.test_mask
        mask_2 = data.val_mask if split == 'val' else data.test_mask
        sample_idxs_1 = torch.where(mask_1)[0]
        sample_idxs_2 = torch.where(mask_2)[0]
        pair_idxs = pair_enumeration(sample_idxs_1.unsqueeze(1), sample_idxs_2.unsqueeze(1)).transpose(0, 1) # (2, sample_size)
        idx1 = pair_idxs[0]
        idx2 = pair_idxs[1]
    elif eval_mode == 'sampling':
        mask_2 = data.val_mask if split == 'val' else data.test_mask
        num_classes = data.y.max().item() + 1
        idx1, idx2 = pair_enumerator.balanced_sampling(max_class_num=num_classes, sample_size=100000, shuffle=False)
    else:
        raise NotImplementedError('Not Implemented Eval Mode:{}'.format(eval_mode))
    
    with torch.no_grad():
        model.eval()
        y_pair = (data.y[idx1] == data.y[idx2]).float().unsqueeze(-1)
        probs_pair, log_probs_clf = model.get_probs_within_domain(data, idx1, idx2, domain=domain)
        pred_pair = (probs_pair > 0.5).long().view(-1)
        # print('#'*10, mode, pred_pair.sum())
        pred_clf =  log_probs_clf[mask_2].max(1)[1]

        if conf_lower_bound is not None:
            assert conf_lower_bound <= 1 and conf_lower_bound >= 0
            high_conf = np.quantile(probs_pair.cpu().numpy(), q=conf_lower_bound)
            low_conf = np.quantile(probs_pair.cpu().numpy(), q=1-conf_lower_bound)
            print('low_conf:{:.4f} | high_conf:{:.4f}'.format(low_conf, high_conf))
            mask_conf = ((probs_pair >= high_conf) + (probs_pair <= low_conf)).view(-1).bool()
            print('Num of high-conf-samples:{}/{}'.format(mask_conf.sum().item(), mask_conf.shape[0]))
        else:
            mask_conf = torch.ones(probs_pair.shape[0]).bool()
        if metric == 'f1':
            score_pair = f1_score(y_pair[mask_conf].long().cpu().numpy(), pred_pair[mask_conf].detach().cpu().numpy(), average='binary')
            score_clf = f1_score(data.y[mask_2].long().cpu().numpy(), pred_clf.detach().cpu().numpy(), average='macro')
        elif metric == 'auc':
            score_pair = roc_auc_score(y_pair[mask_conf].long().cpu().numpy(), pred_pair[mask_conf].detach().cpu().numpy())
            score_clf = roc_auc_score(data.y[mask_2].long().cpu().numpy(), pred_clf.detach().cpu().numpy())
        elif metric == 'acc':
            score_pair = accuracy_score(y_pair[mask_conf].long().cpu().numpy(), pred_pair[mask_conf].detach().cpu().numpy())
            score_clf = accuracy_score(data.y[mask_2].long().cpu().numpy(), pred_clf.detach().cpu().numpy())
        else:
            raise NotImplementedError('NotImplemented Metric:{}'.format(metric))
    return score_pair, score_clf

def eval_adv_v2(data_src, data_tar, model, split='test', metric='f1', enu_list=None, eval_mode='sampling'):
    enu_src, enu_tar, enu_cross = enu_list
    eval_pair_src, eval_clf_src = eval_within_domain_v2(data_src, model, split=split, domain='source', metric=metric, \
                                                     pair_enumerator=enu_src, eval_mode=eval_mode)
    eval_pair_tar, eval_clf_tar = eval_within_domain_v2(data_tar, model, split=split, domain='target', metric=metric, \
                                                     pair_enumerator=enu_tar, eval_mode=eval_mode)
    eval_pair_cross = eval_cross_domain_v2(data_src, data_tar, model, split=split, metric=metric, \
                                        pair_enumerator=enu_cross, eval_mode=eval_mode)
    return (eval_pair_src, eval_clf_src, eval_pair_tar, eval_clf_tar, eval_pair_cross)



def main_adv_v2(args, data_src, data_tar, save=False, repeat=3, num_epoch=200, seed=None, num_layer=2, hidden= 64, metric='f1', \
             use_clf=True, norm_mode='PN', norm_scale=1., eval_per_epoch=1, start_eval_epoch=0, max_class_num=5, \
                sample_size=40000, sim_mode='cosine', backbone='mlp', use_norm=True, eval_mode='sampling', device=None):
    assert device is not None
    data_src = data_src.to(device)
    data_tar = data_tar.to(device)
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
        print('auto fixed data split seed to {}, model init seed to {}'.format(data_split_seed, model_init_seed))

        model = Adversarial_Learner_v2(data_src, data_tar, dim_hidden=hidden, num_layer=2, use_norm=use_norm, \
                                    source_clf=use_clf, norm_mode=norm_mode, norm_scale=norm_scale, sim_mode=sim_mode, backbone=backbone)
        # model.set_device(device)
        model = model.to(device)


        print(data_src, data_tar)
        # build optimizer
        lr = 1e-3
        b1 = 0.5
        b2 = 0.999
        clip_value = 0.01
        optimizer_src_tar = torch.optim.Adam([
            {'params': model.source_learner.parameters(), 'lr': 1e-2, 'weight_decay': 5e-3},
            {'params': model.target_learner.parameters(), 'lr': lr, 'betas': (b1, b2)},
        ])
        # optimizer_AE = torch.optim.Adam(model.target_learner.parameters(), lr=lr, betas=(b1, b2)) 
        optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(b1, b2)) 

        best_acc = {
            'epoch': -1,
            'train': (0,0,0),
            'val': (0,0,0),
            'test': (0,0,0),
            'loss': 666,
        }
        # Train Source-Learner
        pair_enumerator_src_train = Pair_Enumerator(data_src, mode='train')
        pair_enumerator_tar_train = Pair_Enumerator(data_tar, mode='train')
        pair_enumeration_cross_train = Pair_Enumerator_cross(data_src, data_tar, mode='train')

        pair_enumerator_src_val = Pair_Enumerator(data_src, mode='val')
        pair_enumerator_tar_val = Pair_Enumerator(data_tar, mode='val')
        pair_enumeration_cross_val = Pair_Enumerator_cross(data_src, data_tar, mode='val')
        enu_val = (pair_enumerator_src_val, pair_enumerator_tar_val, pair_enumeration_cross_val)

        pair_enumerator_src_test = Pair_Enumerator(data_src, mode='test')
        pair_enumerator_tar_test = Pair_Enumerator(data_tar, mode='test')
        pair_enumeration_cross_test = Pair_Enumerator_cross(data_src, data_tar, mode='test')
        enu_test = (pair_enumerator_src_test, pair_enumerator_tar_test, pair_enumeration_cross_test)

        
        
        
        for epoch in range(1, 1+num_epoch):
            t0 = time.time()
            loss_sim, eval_pair_train, loss_d, loss_ae, loss_g, loss_recons = train_adv_few_shot(epoch, data_src, data_tar, model, optimizer_src_tar, optimizer_D, \
                    pair_enumerator_src_train=pair_enumerator_src_train, pair_enumerator_tar_train=pair_enumerator_tar_train, metric=metric, \
                    pair_enumerator_cross_train=pair_enumeration_cross_train, max_class_num=max_class_num, sample_size=sample_size, use_clf=use_clf)

            print(''.format(loss_g, loss_recons))
            log_ae = '[AE]Epoch: {:03d}, Loss_ae:{:.4f} | Loss_recons:{:.4f} | Loss_g:{:.4f} | Loss_d:{:.4f}  Time(s/epoch):{:.4f}'.format(
                epoch, loss_ae, loss_recons, loss_g, loss_d, time.time() - t0)
            print(log_ae)
            
            # val_pair, val_clf = eval_sim(data_src, model.source_learner, metric, pair_enumerator_val, data_src.val_mask, max_class_num=2, sample_size=20000, use_clf=use_clf)
            # test_pair, test_clf = eval_sim(data_src, model.source_learner, metric, pair_enumerator_test, data_src.test_mask, max_class_num=2, sample_size=20000, use_clf=use_clf)
            if epoch >= start_eval_epoch and epoch % eval_per_epoch == 0:
                eval_pair_src_val, eval_clf_src_val, eval_pair_tar_val, eval_clf_tar_val, eval_pair_cross_val = eval_adv_v2(data_src, \
                    data_tar, model, split='val', metric=metric, enu_list=enu_val, eval_mode=eval_mode)
                eval_pair_src_test, eval_clf_src_test, eval_pair_tar_test, eval_clf_tar_test, eval_pair_cross_test = eval_adv_v2(data_src, \
                    data_tar, model, split='test', metric=metric, enu_list=enu_test, eval_mode=eval_mode)

                if use_clf:
                    log_source = '[Sim]Epoch: {:03d}, Loss:{:.4f} | Train Pair:{:.4f}/{:.4f}/{:.4f} | Val Pair:{:.4f}/{:.4f}/{:.4f} | Test Pair:{:.4f}/{:.4f}/{:.4f} | Val CLF:{:.4f}/{:.4f} | Test CLF:{:.4f}/{:.4f} | Time(s/epoch):{:.4f}'.format(\
                        epoch, loss_sim, *eval_pair_train, *(eval_pair_src_val, eval_pair_tar_val, eval_pair_cross_val), *(eval_pair_src_test, eval_pair_tar_test, eval_pair_cross_test), \
                        eval_clf_src_val, eval_clf_tar_val, eval_clf_src_test, eval_clf_tar_test, time.time() - t0)
                else:
                    log_source = '[Sim]Epoch: {:03d}, Loss:{:.4f} | Train Pair:{:.4f}/{:.4f}/{:.4f} | Val Pair:{:.4f}/{:.4f}/{:.4f} | Test Pair:{:.4f}/{:.4f}/{:.4f}| Time(s/epoch):{:.4f}'.format(\
                        epoch, loss_sim, *eval_pair_train, *(eval_pair_src_val, eval_pair_tar_val, eval_pair_cross_val), *(eval_pair_src_test, eval_pair_tar_test, eval_pair_cross_test), time.time() - t0)
                print(log_source)
                # if  eval_pair_cross_val > best_acc['val'][2] and loss_sim < best_acc['loss']:
                if eval_pair_cross_val > best_acc['val'][2]:
                # if loss_sim < best_acc['loss']:
                    best_acc['train'] = eval_pair_train
                    best_acc['val'] = (eval_pair_src_val, eval_pair_tar_val, eval_pair_cross_val)
                    best_acc['test'] = (eval_pair_src_test, eval_pair_tar_test, eval_pair_cross_test)
                    best_acc['loss'] = loss_sim
                    best_acc['epoch'] = epoch
                    if save:
                        torch.save(model.state_dict(), f'../ckpt/model_AdvLearner_{args.dataset_name}_best.ckpt')
            # # Train Target-Learner
            # t0 = time.time()
            # loss_train = train_ae(data_tar, model.target_learner, optimizer_AE)
            # # eval_res = test(data, model, dataset_name=dataset_name, metric=metric)
            # log = 'Epoch: {:03d}, Loss:{:.4f}  Time(s/epoch):{:.4f}'.format(epoch, loss_train, time.time() - t0)
            # print(log)
        if save:
            torch.save(model.state_dict(), f'../ckpt/model_AdvLearner_{args.dataset_name}_final.ckpt')   
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
