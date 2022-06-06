import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from tqdm import tqdm
import numpy as np
import ipdb

import utils
import pretrain_utils
import models
import layers

def fuse_feature(feature_list, fuse='last'): #get embedding feature from encoder

    if fuse == 'last':
        fused_feat = feature_list[-1]
    elif fuse == 'avg':
        fused_feat = torch.mean(torch.stack(feature_list))
    elif fuse == 'concat':
        fused_feat = torch.cat(feature_list, dim=-1)

    return fused_feat

def cal_feat_dim(args): # get dimension of obtained embedding feature
    emb_dim = args.nhid
    if args.fuse == 'concat':
        emb_dim = emb_dim * args.enc_layer

    return emb_dim


class Trainer(object):
    def __init__(self, args, model, weight):#
        self.args = args

        self.in_dim = cal_feat_dim(args)
        self.loss_weight = weight
        self.models = []
        self.models.append(model)
        if args.model=='DISGAT':
            if args.residue:
                self.fuse1 = layers.FuseLayer(args, args.nhead, nfeat=args.nhid, residue=args.size)
                self.fuse2 = layers.FuseLayer(args, args.nhead, nfeat=args.nhid, residue=args.nhid)
            else:
                self.fuse1 = layers.FuseLayer(args, args.nhead, nfeat=args.nhid)
                self.fuse2 = layers.FuseLayer(args, args.nhead, nfeat=args.nhid)


            if args.cuda:
                self.fuse1.cuda()
                self.fuse2.cuda()
            self.models.append(self.fuse1)
            self.models.append(self.fuse2)

        self.models_opt = []
        for model in self.models:
            self.models_opt.append(optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay))

    def train_step(self, data, pre_adj):#pre_adj corresponds to adj used for generating ssl signal
        raise NotImplementedError('train not implemented for base class')

    def inference(self, data):
        raise NotImplementedError('infer not implemented for base class')

    def get_label_all(self, feature, adj):
        raise NotImplementedError('get_label_all not implemented for base class')

    def get_em(self, feature, adj):
        if self.args.model != 'DISGAT':
            output = self.models[0].get_em(feature, adj)
        else:
            fusers = [self.fuse1, self.fuse2]
            output = self.models[0].get_em(feature, adj, fusers)

        output = fuse_feature(output, fuse=self.args.fuse)

        return output

    def analyze_disentangle(self,feature,adj):
        # for analyze learned distribution of disentangled edge-aware channels
        #at_distance_list: [nlayers*1],obtain attention KL distance at each layer
        #correlation_graph_lists: [nlayers * (nheads*nheads)],obtain correlation between attention heads as grid graph at each layer
        #feat_dim_graph_lists = [nlayers * (nhid *nhid) ], obtain correlation between each dim pair as grid graph of each layer
        assert self.args.model == 'DISGAT', 'analyze disentanglement is only implemented for DISGAT'
        fusers = [self.fuse1, self.fuse2]

        feats = self.models[0].get_em(feature, adj, fusers) # nlayers * (n_node*nhid)

        #edge indices for comparing distribution of disentangled edges
        if True:
            if adj.is_sparse:
                adj_cand = adj.to_dense()
            else:
                adj_cand = adj
            adj_cand = (adj_cand!=0).int()
            label = (adj_cand!=0).float()
            mask = label.new(label.shape).fill_(0)
            #randomly select edge entries for training
            edge_ratio = adj_cand.sum()/(adj.shape[0]*adj.shape[0])
            random_ini_mask = (torch.rand(size=mask.shape) < edge_ratio.item()*3).int()
            mask[:] = random_ini_mask[:]
        
            indices = adj_cand.nonzero().cpu().numpy()
            edge_num = int(indices.shape[0]//3)
            np.random.shuffle(indices)
            pos_indices = indices[:edge_num]
            mask[pos_indices[:,0],pos_indices[:,1]] = 1

            indices = mask.nonzero().transpose(0,1) #(2,edge_num)

            if adj.is_sparse:
                adjs = self.models[0].predict_adjs_sparse(feature, adj, fusers, auxiliary_edges=indices) #nlayer *nhead*1 * (n_edges*1)
            else:
                adjs = self.models[0].get_adjs(feature, adj, fusers)

        ##compute the metrics
        at_cor_graph_lists = []
        at_distance_list = []
        feat_cor_graph_lists = []

        for layer in range(2):
            feat_cor_graph_lists.append(utils.group_correlation(feats[layer].transpose(0,1)))

            adj_layer = [adj[0] for adj in adjs[layer]]
            adj_layer = torch.stack(adj_layer).squeeze()
            at_cor_graph =utils.group_correlation(adj_layer)
            at_cor_graph_lists.append(at_cor_graph)
            
            at_distance_list.append(torch.mean(torch.abs(at_cor_graph)).item())

        return at_distance_list, at_cor_graph_lists, feat_cor_graph_lists

    def reg_fuser(self):
        l1_norm1 = sum(p.abs().sum() for p in self.fuse1.parameters())
        l1_norm2 = sum(p.abs().sum() for p in self.fuse2.parameters())
        reg_loss = self.args.reg_weight * (l1_norm1+l1_norm2)
        #print('reg_loss: {}'.format(reg_loss.item()))

        return  reg_loss

    
    def train_batch(self, graphset, label, batch_id):#batch_id: indicating the batches being used
        raise NotImplementedError('train_batch not implemented for base class')
        

#Node classification
class ClsTrainer(Trainer):
    def __init__(self, args, model, labels, weight=1.0):
        super().__init__(args, model, weight)        

        
        self.classifier = models.MLP(in_feat=self.in_dim, hidden_size=args.nhid, out_size=labels.max().item() + 1, layers=args.cls_layer)
        if args.cuda:
            self.classifier.cuda()
        self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
        self.models.append(self.classifier)
        self.models_opt.append(self.classifier_opt)

        #split train, test, valid for node classification
        self.idx_train, self.idx_val, self.idx_test, self.class_num_mat = utils.split(labels, train_ratio=args.node_sup_ratio)
        if args.cuda:
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()

        '''
        tmp = self.idx_train
        self.idx_train = self.idx_test
        self.idx_test = tmp
        '''

        #print(self.class_num_mat)

    def train_step(self, data, labels, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        feature, adj = data
        
        output = self.get_em(feature,adj)
        output = self.models[-1](output, cls=True)

        #ipdb.set_trace()
        loss_log = F.nll_loss(output[self.idx_train], labels[self.idx_train])
        acc_train = utils.accuracy(output[self.idx_train], labels[self.idx_train])
        loss_train = loss_log
        if self.args.model=='DISGAT':
            reg_log = self.reg_fuser()
        else:
            reg_log = loss_log
        if self.args.reg:
            loss_train = loss_log + reg_log

        loss_train = loss_train*self.loss_weight
        loss_train.backward()

        for opt in self.models_opt:
            opt.step()
        
        #ipdb.set_trace()

        loss_val = F.nll_loss(output[self.idx_val], labels[self.idx_val])
        acc_val = utils.accuracy(output[self.idx_val], labels[self.idx_val])
        #utils.print_class_acc(output[self.idx_val], labels[self.idx_val], self.class_num_mat[:,1])
        roc_val, macroF_val = utils.Roc_F(output[self.idx_val], labels[self.idx_val], self.class_num_mat[:,1])

        
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_log.item()),
              'loss_reg: {:.4f}'.format(reg_log.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()))
        
        log_info = {'loss_train': loss_log.item(), 'acc_train': acc_train.item(), 'loss_reg': reg_log.item(),
                     'loss_val': loss_val.item(), 'acc_val': acc_val.item(), 'roc_val': roc_val, 'macroF_val': macroF_val }

        return log_info

    def train_batch(self, graphset, label, batch_id):
        if (batch_id+1)*self.args.batch_size >= len(self.idx_train):
            raise ValueError("idx out of boundary")

        idx_used = np.arange(batch_id*self.args.batch_size, (batch_id+1)*self.args.batch_size)
        idx_batch = self.idx_train[idx_used]

        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        feature, adj, _ = graphset.get_batch(idx_batch, self.args.SubgraphSize)
        

        output = self.get_em(feature,adj)
        output = self.models[-1](output, cls=True)

        loss_train = F.nll_loss(output[:,0], label[idx_batch]) * self.loss_weight
        acc_train = utils.accuracy(output[:,0], label[idx_batch])

        if self.args.reg:
            loss_train = loss_train + self.reg_fuser()

        loss_train.backward()

        for opt in self.models_opt:
            opt.step()

        log_info = {'loss_train': loss_train.item(), 'acc_train': acc_train.item()}
        
        print('batch: {:05d}'.format(batch_id),
              'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()))

        return log_info

    def test_batch(self, graphset, label, is_test=True):
        
        loss = []
        acc = []
        for i, model in enumerate(self.models):
            model.eval()
        if is_test:
            idx_used = self.idx_test
        else:
            idx_used = self.idx_val

        for i in range(len(idx_used)//self.args.batch_size):
            idx_batch = idx_used[i*self.args.batch_size:(i+1)*self.args.batch_size]

            feature, adj, _ = graphset.get_batch(idx_batch, self.args.SubgraphSize)
        

            output = self.get_em(feature,adj)
            output = self.models[-1](output, cls=True)


            #ipdb.set_trace()
            loss_test = F.nll_loss(output[:,0], label[idx_batch])
            acc_test = utils.accuracy(output[:,0], label[idx_batch])

            loss.append(loss_test.item())
            acc.append(acc_test.item())

        log_info = {'loss_test': sum(loss)/len(loss), 'acc_test': sum(acc)/len(acc)}
        
        print("Test set results:",
              "loss= {:.4f}".format(sum(loss)/len(loss)),
              "accuracy= {:.4f}".format(sum(acc)/len(acc)))

        return log_info

    def test(self, data, labels, epoch = 0):
        for i, model in enumerate(self.models):
            model.eval()

        feature, adj = data
        
        output = self.get_em(feature, adj)

        output = self.models[-1](output, cls=True)

        loss_test = F.nll_loss(output[self.idx_test], labels[self.idx_test])
        acc_test = utils.accuracy(output[self.idx_test], labels[self.idx_test])

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        utils.print_class_acc(output[self.idx_test], labels[self.idx_test], self.class_num_mat[:,2], pre='test')
        
        roc_test, macroF_test = utils.Roc_F(output[self.idx_test], labels[self.idx_test], self.class_num_mat[:,2])
        
        log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'roc_test': roc_test, 'macroF_test': macroF_test}

        return log_info


#Edge prediction
class EdgeTrainer(Trainer):
    def __init__(self, args, model, labels, weight=1.0):
        super().__init__(args, model, weight)        

        self.classifier = models.EdgePredictor(nfeat=self.in_dim, nhid=args.nhid, layers=args.EdgePred_layer)

        if args.cuda:
            self.classifier.cuda()

        self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.models.append(self.classifier)
        self.models_opt.append(self.classifier_opt)

        #split train, test, valid for node classification
        self.idx_train, self.idx_val, self.idx_test, self.class_num_mat = utils.split(labels, train_ratio=0.5)
        if args.cuda:
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()

    def train_step(self, data, labels, epoch):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        feature, adj = data
        if adj.is_sparse:
            adj_tgt = adj.to_dense()
        else:
            adj_tgt = adj
        adj_tgt = (adj_tgt !=0).float()

        output = self.get_em(feature, adj)
        output = self.models[-1](output)

        loss_train = utils.adj_mse_loss(output[self.idx_train,:][:,self.idx_train], adj_tgt[self.idx_train,:][:,self.idx_train])*self.loss_weight
        if self.args.reg:
            loss_train = loss_train + self.reg_fuser()
        loss_train.backward()

        for opt in self.models_opt:
            opt.step()

        loss_val = utils.adj_mse_loss(output[self.idx_val,:][:,self.idx_val], adj_tgt[self.idx_val,:][:,self.idx_val])*self.loss_weight

        output[output>0.5] = 1.0
        output[output<0.5] = 0.0
        acc_train, TP_train, TN_train = utils.adj_accuracy(output[self.idx_train,:][:,self.idx_train], adj_tgt[self.idx_train,:][:,self.idx_train])
        acc_val, TP_val, TN_val = utils.adj_accuracy(output[self.idx_val,:][:,self.idx_val], adj_tgt[self.idx_val,:][:,self.idx_val])

        '''
        print('Epoch: {:05d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'TP_train: {:.4f}'.format(TP_train.item()),
            'TP_val: {:.4f}'.format(TP_val.item()),
            'TN_train: {:.4f}'.format(TN_train.item()),
            'TN_val: {:.4f}'.format(TN_val.item()))
        '''
        
        log_info = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                     'loss_val': loss_val.item(), 'acc_val': acc_val.item(),
                     'TP_train': TP_train.item(), 'TP_val': TP_val.item(),
                     'TN_train': TN_train.item(), 'TN_val': TN_val.item()}

        return log_info


    def test(self, data, labels, epoch = 0):
        for i, model in enumerate(self.models):
            model.eval()

        feature, adj = data
        if adj.is_sparse:
            adj_tgt = adj.to_dense()
        else:
            adj_tgt = adj
        adj_tgt = (adj_tgt !=0).float()

        output = self.get_em(feature, adj)
        output = self.models[-1](output)

        loss_test = utils.adj_mse_loss(output[self.idx_test,:], adj_tgt[self.idx_test,:])
        
        output[output>0.5] = 1.0
        output[output<0.5] = 0.0
        acc_test, TP_test, TN_test = utils.adj_accuracy(output[self.idx_test,:], adj_tgt[self.idx_test,:])


        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()),
              "TP= {:.4f}".format(TP_test.item()),
              "TN= {:.4f}".format(TN_test.item()))

        log_info = {'loss_test': loss_test.item(), 'acc_test': acc_test.item(), 'TP_test': TP_test.item(), 'TN_test': TN_test.item()}

        return log_info