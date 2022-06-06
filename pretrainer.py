
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
from trainer import Trainer, fuse_feature
import dataset


#predict attributes for masked nodes
class PredTrainer(Trainer):
    def __init__(self, args, model, weight):
        super().__init__(args, model, weight)

        self.masked_ratio = 0.05
        
        self.classifier = models.MLP(in_feat=self.in_dim, hidden_size=args.nhid, out_size=args.pretext_dim)
        if args.cuda:
            self.classifier.cuda()

        self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.models.append(self.classifier)
        self.models_opt.append(self.classifier_opt)

        self.criterion = nn.MSELoss()

    def get_label_all(self, feature, adj=None):
        pretext = pretrain_utils.obtain_pretext(feature, self.args.pretext_dim)
        #pretext = torch.from_numpy(pretext)
        pretext = feature.new_tensor(pretext)

        return pretext

    def train_step(self, data, label, pre_adj=None):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        feature, adj = data
        sampled_idx = np.random.choice(feature.shape[0], size = int(feature.shape[0]*self.masked_ratio), replace=False)
        
        train_ind = feature.new_tensor(sampled_idx).long()

        cached_feature = feature.clone()
        cached_feature[train_ind] = 0.0

        predicted = self.inference([cached_feature, adj], train_ind)
        loss = self.criterion(predicted, label[train_ind])
        loss_log = loss
        if self.args.reg:
            reg_log = self.reg_fuser()
        else:
            reg_log = loss_log

        if self.args.reg:
            loss = loss + reg_log
        loss = loss * self.loss_weight
        loss.backward()

        for opt in self.models_opt:
            if self.loss_weight != 0:
                opt.step()
        
        print('Attribute prediction loss : {}, reg loss: {}'.format(loss_log.item(), reg_log.item()))

        log_info = {'loss_train': loss_log.item(), 'loss_reg': reg_log.item()}

        return log_info

    def inference(self, data, ind=None):#predict attributes for ind
        feature, adj = data

        if ind is None:
            ind = feature.new_tensor(list(np.arange(feature.shape[0]))).long()

        embed = self.get_em(feature, adj)
        context_embed = pretrain_utils.obtain_context(embed, adj, ind)

        predicted = self.models[-1](context_embed)

        return predicted

    
    def train_batch(self, graphset, label, batch_id):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        sampled_idx = np.random.choice(label.shape[0], size = self.args.batch_size, replace=False)
        feature, adj,_ = graphset.get_batch(sampled_idx, preset_size=self.args.SubgraphSize)
        feature[:,0] = 0.0

        predicted = self.inference([feature, adj], sampled_idx)
        loss = self.criterion(predicted[:,0], label[sampled_idx])
        loss_log = loss
        if self.args.reg:
            reg_log = self.reg_fuser()
        else:
            reg_log = loss_log
        if self.args.reg:
            loss = loss + reg_log
        loss = loss * self.loss_weight
        loss.backward()

        for opt in self.models_opt:
            if self.loss_weight != 0:
                opt.step()

        
        print('Attribute prediction loss : {}, reg loss: {}'.format(loss_log.item(), reg_log.item()))

        log_info = {'loss_train': loss_log.item(), 'loss_reg':reg_log.item()}

        return log_info






#contrastive learning between node and neighborhood
#class GraphInfomaxTrainer(Trainer):



#predict node pair-wise distances on graph
class DistanceTrainer(Trainer): 
    def __init__(self, args, model, weight):
        super().__init__(args, model, weight)

        if self.args.dataset == 'cora_full':
            self.distance_thresh = [1,2,3]
        if self.args.dataset == 'squirrel':
            self.distance_thresh = [1,2,3]
        if self.args.dataset == 'chameleon':
            self.distance_thresh = [1,2,3]
        if self.args.dataset == 'deezer':
            self.distance_thresh = [1,2,3]
        #below are not tested
        if self.args.dataset == 'imdb':
            self.distance_thresh = [1,2,3]
        if self.args.dataset == 'dblp':
            self.distance_thresh = [1,2,3]
        if self.args.dataset == 'arxiv':
            self.distance_thresh = [1,2,3]
        
        self.train_ratio = 0.01
        self.anchor_number = 24

        self.classifier = models.MLP(in_feat=self.in_dim*2, hidden_size=args.nhid*2, out_size=len(self.distance_thresh)+1)
        if args.cuda:
            self.classifier.cuda()
        self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.models.append(self.classifier)
        self.models_opt.append(self.classifier_opt)

        self.criterion = nn.NLLLoss()

    
    def get_label(self, adj, anchor_id, sampled_idx):#anchor_id&adj: tensor; sampled_idx: numpy array
        #return: numpy array, distance
        pair_distance = pretrain_utils.compute_dist_to_group(adj, anchor_id)[sampled_idx,:]
        pair_distance = self.post_dist(pair_distance)

        #change to a 1-D array
        pair_distance = np.reshape(pair_distance, -1, order='F')

        return pair_distance

    def get_label_all(self, feature, adj):#anchor_id&adj: tensor; sampled_idx: numpy array
        #return: numpy array, distance
        #check if generated
        if not os.path.exists('./resource/{}'.format(self.args.dataset)):
            os.makedirs('./resource/{}'.format(self.args.dataset))

        path_distance = './resource/{}/ComputedPath_edge{}.npy'.format(self.args.dataset, self.args.used_edge)
        if os.path.exists(path_distance):
            pair_distance = np.load(path_distance)
        else:
            pair_distance = pretrain_utils.compute_dist_all(adj)
            pair_distance = self.post_dist(pair_distance)
            np.save(path_distance, pair_distance)

        return pair_distance

    def post_dist(self, distance):
        
        selected_inds=[]
        thresh_last = -99
        for thresh in self.distance_thresh:
            selected_ind = (distance<=thresh) & (distance > thresh_last)
            selected_inds.append(selected_ind)
            thresh_last = thresh
        selected_inds.append(distance>thresh)

        for i, inds in enumerate(selected_inds):
            distance[inds] = i

        return distance


    def train_step(self, data, dist_matrix=None, pre_adj=None):#dist_matrix should be numpy
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        feature, adj = data
        sampled_idx = np.random.choice(feature.shape[0], size = int(feature.shape[0]*self.train_ratio), replace=False)
        train_ind = feature.new_tensor(sampled_idx).long()

        #obtain labels.
        sampled_anchor = np.random.choice(feature.shape[0], size = self.anchor_number, replace=False)
        anchor_id = feature.new_tensor(sampled_anchor).long()

        if dist_matrix is None:
            if pre_adj is None:
                pre_adj = adj
            gt_distance = self.get_label(pre_adj, anchor_id, sampled_idx)
        else:
            gt_distance = dist_matrix[sampled_idx,:][:,sampled_anchor]
            gt_distance = gt_distance.transpose().reshape(-1)
        
        gt_distance = feature.new_tensor(gt_distance).long()

        predicted = self.inference([feature, adj], train_ind, anchor_id)
        loss = self.criterion(predicted, gt_distance)
        loss_log = loss
        if self.args.reg:
            reg_log = self.reg_fuser()
        else:
            reg_log = loss_log
        if self.args.reg:
            loss = loss + reg_log
        loss = loss * self.loss_weight

        loss.backward()

        for opt in self.models_opt:
            if self.loss_weight != 0:
                opt.step()

        print('Distance prediction loss : {}, reg loss: {}'.format(loss_log.item(), reg_log.item()))

        log_info = {'loss_train': loss_log.item(), 'loss_reg': reg_log.item()}

        return log_info

    def inference(self, data, train_ind, anchor_ind):#predict attributes for ind
        feature, adj = data

        embed = self.get_em(feature, adj)

        target_features = embed[train_ind]
        anchor_features = embed[anchor_ind]

        con_feat = torch.cat([torch.cat((target_features, anchor_features[i,:].expand_as(target_features)), dim=-1) for i in range(anchor_ind.shape[0])], dim=0)
        predicted = self.models[-1](con_feat, cls=True)

        return predicted


#predict context distribution
class ContextPredTrainer(Trainer):
    def __init__(self, args, model, weight):
        super().__init__(args, model, weight)

        self.classifier = models.MLP(in_feat=self.in_dim, hidden_size=args.nhid, out_size=args.cluster_num)
        if args.cuda:
            self.classifier.cuda()
        self.classifier_opt = optim.Adam(self.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.models.append(self.classifier)
        self.models_opt.append(self.classifier_opt)

        self.criterion = nn.MSELoss()

    def get_label_all(self, feature, adj):
        pretext = pretrain_utils.obtain_pretext(feature, self.args.pretext_dim)
        pretext_dist = pretrain_utils.context_dist_all(pretext, adj, self.args.cluster_num)

        #context = torch.from_numpy(pretext_dist)
        context = feature.new_tensor(pretext_dist)

        return context

    def train_step(self, data, label, pre_adj=None):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()


        predicted = self.inference(data)
        loss = self.criterion(predicted, label)
        loss_log = loss
        if self.args.reg:
            reg_log = self.reg_fuser()
        else:
            reg_log = loss_log
        if self.args.reg:
            loss = loss + reg_log
        loss = loss * self.loss_weight

        loss.backward()

        for opt in self.models_opt:
            if self.loss_weight != 0:
                opt.step()
        
        
        print('Context prediction loss : {}, reg loss: {}'.format(loss_log.item(), reg_log.item()))

        log_info = {'loss_train': loss_log.item(),'loss_reg': reg_log.item()}

        return log_info

    def inference(self, data, ind=None):#predict attributes for ind
        feature, adj = data

        if ind == None:
            ind = feature.new_tensor(list(np.arange(feature.shape[0])))

        embed = self.get_em(feature, adj)

        predicted = self.models[-1](embed)

        return predicted

    
    def train_batch(self, graphset, label, batch_id):
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        sampled_idx = np.random.choice(label.shape[0], size = self.args.batch_size, replace=False)
        feature, adj,_ = graphset.get_batch(sampled_idx, preset_size=self.args.SubgraphSize)


        predicted = self.inference((feature, adj))
        loss = self.criterion(predicted[:,0], label[sampled_idx])
        loss_log = loss
        if self.args.reg:
            reg_log = self.reg_fuser()
        else:
            reg_log = loss_log
        if self.args.reg:
            loss = loss + reg_log

        loss = loss * self.loss_weight

        loss.backward()

        for opt in self.models_opt:
            if self.loss_weight != 0:
                opt.step()
        
        
        print('Context prediction loss : {}, reg loss: {}'.format(loss_log.item(), reg_log.item()))

        log_info = {'loss_train': loss_log.item(), 'loss_reg': reg_log.item()}

        return log_info

#obtain pos/neg supervision for edges captured by different heads
class GeneratedEdgeTrainer(Trainer): 
    #currently can only run under dense setting. For sparse setting, computation of attention for negative samples would be difficult
    #use sampling-based training
    def __init__(self, args, model, weight):
        super().__init__(args, model, weight)
        self.dis_type = args.dis_type #1: homo/hetero disentanglement; 2: class-wise disentanglement
        self.criterion = nn.MSELoss()
        self.constrain_layer = args.constrain_layer#0 for all, 1 for the first, 2 for the second
        self.dis_adjs = [] #list of adj tensors. should be dense
        self.labels = None
        self.args = args

        self.sparse = args.sparse

    def get_label_all(self, feature, adj, labels, load=True):#anchor_id&adj: tensor; sampled_idx: numpy array
        #return: list of torch tensors
        #check if generated

        if not os.path.exists('./resource/{}'.format(self.args.dataset)):
            os.makedirs('./resource/{}'.format(self.args.dataset))

        file_path = './resource/{}/DisEdges.pt'.format(self.args.dataset)
        self.labels = labels
        
        dis_adjs = []
        if os.path.exists(file_path) and load and not self.args.conformT:
            dis_edges = torch.load(file_path)
            for i in range(dis_edges.shape[0]):
                a_new = dis_edges[i,:]
                a_new = feature.new(a_new.numpy())
                dis_adjs.append(a_new)
        elif not self.args.conformT:#take all edge heterophily as known
            #get adjs in tensor, form list
            '''
            if adj.is_sparse:
                indices = adj.coalesce().indices()
                if self.dis_type == 1:
                    src_labels = labels[indices[0, :]]
                    tgt_labels = labels[indices[1, :]]
                
                    values = adj.coalesce().values()
                    values[src_labels==tgt_labels] =1
                    values[src_labels!=tgt_labels] =0
                    dis_adjs.append(values)
                    
                    values = adj.coalesce().values()
                    values[src_labels==tgt_labels] =0
                    values[src_labels!=tgt_labels] =1
                    dis_adjs.append(values)

                elif self.dis_type == 2:
                    src_labels = labels[indices[0, :]]
                    tgt_labels = labels[indices[1, :]]

                    for cls in range(self.args.nclass):                    
                        values = adj.coalesce().values()
                        values[:] = 0

                        chosen_ind = torch.logical_and(src_labels == cls, src_labels == tgt_labels)
                        values[chosen_ind] = 1
                        dis_adjs.append(values)

                else:
                    print('not implemented yet')
                    ipdb.set_trace()
            '''

            #implemented for dense graph
            adj = adj.to_dense()

            #remove diagonal elements
            #ind = np.diag_indices(adj.shape[0])
            #adj[ind[0], ind[1]] =  adj.new(adj.shape[0]).fill_(0)

            if not adj.is_sparse:

                if self.dis_type ==1:
                    homo_mask = adj.new(adj.shape).fill_(0)

                    homo_mask = ((labels.unsqueeze(0).expand(adj.shape))==(labels.unsqueeze(-1).expand(adj.shape))).int()

                    hetero_mask = 1-homo_mask
                    adj_ind = (adj!=0).int()
                    dis_adjs.append(((homo_mask+adj_ind)==2).float())
                    dis_adjs.append(((hetero_mask+adj_ind)==2).float())

                else:
                    print('not implemented yet')
                    ipdb.set_trace()

            #save
            dis_edges = torch.stack(dis_adjs, dim=0)
            torch.save(dis_edges.cpu(), file_path)

        else:#real setting, not using pseudo label,

            idx_train, idx_val, idx_test, class_num_mat = utils.split(labels, train_ratio=self.args.node_sup_ratio)
            
            #implemented for dense graph
            adj = adj.to_dense()

            #remove diagonal elements
            #ind = np.diag_indices(adj.shape[0])
            #adj[ind[0], ind[1]] =  adj.new(adj.shape[0]).fill_(0)

            if not adj.is_sparse:

                if self.dis_type ==1:
                    labeled_nodes = labels.new(labels.shape).fill_(0)
                    idx_known = torch.cat((idx_train, idx_val),dim=-1)
                    labeled_nodes[idx_known] = 1
                    labeled_edges = (labeled_nodes.unsqueeze(0).expand(adj.shape)*labeled_nodes.unsqueeze(1).expand(adj.shape)).int()

                    homo_mask = adj.new(adj.shape).fill_(0)
                    homo_mask = ((labels.unsqueeze(0).expand(adj.shape))==(labels.unsqueeze(-1).expand(adj.shape))).int()
                    homo_mask = (homo_mask+labeled_edges == 2).int()

                    #ind = np.diag_indices(adj.shape[0])
                    #homo_mask[ind[0], ind[1]] =  adj.new(adj.shape[0]).fill_(1)

                    hetero_mask = ((labels.unsqueeze(0).expand(adj.shape))!=(labels.unsqueeze(-1).expand(adj.shape))).int()
                    hetero_mask = (hetero_mask+labeled_edges == 2).int()


                    adj_ind = (adj!=0).int()
                    dis_adjs.append(((homo_mask+adj_ind)==2).float())
                    dis_adjs.append(((hetero_mask+adj_ind)==2).float())

                else:
                    print('not implemented yet')
                    ipdb.set_trace()

            #save
            dis_edges = torch.stack(dis_adjs, dim=0)
            #torch.save(dis_edges.cpu(), file_path)

        for i, disedge in enumerate(dis_adjs):
            print('edge group {} for edge disentanglement SSL size: {}'.format(i, disedge.sum()))    

        self.dis_adjs = dis_adjs

        return self.dis_adjs

    def update_adjs(self, feature, adj, labels):
        #use pseudo labels to update edge set
        print('not implememnted yet')
        ipdb.set_trace()
        #need to consider confidence, and left-out edges
        self.get_label_all(feature,adj,labels,load=False) #update adjs with pseudo labels

        return

    def sample_train(self):
        # if dense: return: adj_labels: [(node*node),(node*node)] ,list of adjs as supervision on disentanglement
        #                   adj_masks: [(node*node),(node*node)], list of masks, to filther out not-used edges
        # if sparse: return: adj_labels: [sampled_edge_num, sampled_Edge_num], contains supervision on each disentanglement
        #                    adj_masks: [sampled_edge_num, sampled_Edge_num], all 1

        adj_labels = []
        adj_masks = []

        for adj in self.dis_adjs:
            if not self.args.sparse:
                label = (adj!=0).float()
                mask = label.new(label.shape).fill_(0)

                #randomly select edge entries for training
                edge_ratio = adj.sum()/(adj.shape[0]*adj.shape[0])
                random_ini_mask = (torch.rand(size=mask.shape) < edge_ratio.item()*3).int()
                mask[:] = random_ini_mask[:]

                #select positive edge entries
                indices = adj.nonzero().cpu().numpy()
                edge_num = int(indices.shape[0]//3)
                np.random.shuffle(indices)
                pos_indices = indices[:edge_num]
                mask[pos_indices[:,0],pos_indices[:,1]] = 1

                adj_labels.append(label)
                adj_masks.append(mask)
            else:

                label = (adj!=0).float()

                mask = label.new(label.shape).fill_(0)
                #randomly select edge entries for training
                edge_ratio = adj.sum()/(adj.shape[0]*adj.shape[0])
                random_ini_mask = (torch.rand(size=mask.shape) < edge_ratio.item()*3).int()
                mask[:] = random_ini_mask[:]

                #select positive edge entries
                indices = adj.nonzero().cpu().numpy()
                edge_num = int(indices.shape[0]//3)
                np.random.shuffle(indices)
                pos_indices = indices[:edge_num]
                mask[pos_indices[:,0],pos_indices[:,1]] = 1

                #get indices for sparse inference
                indices = mask.nonzero().transpose(0,1) #(2,edge_num)
                label_used = label[indices[0],indices[1]]

                adj_labels.append(label_used)
                adj_masks.append(indices)

        return adj_labels, adj_masks

    def train_step(self, data, pre_adjs=None):#pre_adj is a list containing weak sup over adjs
        #pre_adjs is not used, use self.dis_adjs instead
        
        pre_adjs = self.dis_adjs#
        assert type(pre_adjs)==list, "weak supervision on decoupled adjs must be a list!"
        assert self.dis_type == 1, "currently only use homo&hetero edge disentanglement"

        
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        # sampling training pairs for homo&contrastive, hetero&contrastive
        adj_labels, adj_masks = self.sample_train()

        #compute loss
        loss = None
        pred_adjs = self.inference(data, adj_masks) #[[A11, A12, ...], [A21, A22, ...]] of size (layer*nhead*node*node). Contains dense edge
        for i, pred_adj in enumerate(pred_adjs):
            if self.constrain_layer==0 or self.constrain_layer==i:
                #sup on homo part
                N = len(pred_adj)

                if not self.sparse:
                    pred_homo = torch.sigmoid(torch.sum(torch.stack(pred_adj[:int(N/2)]), dim=0))
                    pred_hetero = torch.sigmoid(torch.sum(torch.stack(pred_adj[int(N/2):]), dim=0))

                    if loss is None:
                        loss = utils.masked_adj_mse_loss(pred_homo, adj_labels[0], adj_masks[0])
                    else:
                        loss = loss +  utils.masked_adj_mse_loss(pred_homo, adj_labels[0], adj_masks[0])

                    loss = loss +  utils.masked_adj_mse_loss(pred_hetero, adj_labels[1], adj_masks[1])

                else:
                    homo_pred = []
                    hetero_pred = []
                    for head in pred_adj:
                        homo_pred.append(head[0])
                        hetero_pred.append(head[1])

                    pred_homo = torch.sigmoid(torch.sum(torch.stack(homo_pred[:int(N/2)]), dim=0))
                    pred_hetero = torch.sigmoid(torch.sum(torch.stack(hetero_pred[int(N/2):]), dim=0))

                    if loss is None:
                        loss = utils.adj_mse_loss(pred_homo.squeeze(), adj_labels[0])
                    else:
                        loss = loss +  utils.adj_mse_loss(pred_homo.squeeze(), adj_labels[0])

                    loss = loss +  utils.adj_mse_loss(pred_hetero.squeeze(), adj_labels[1])

        loss_log = loss
        loss = loss * self.loss_weight
        loss.backward()

        for opt in self.models_opt:
            if self.loss_weight != 0:
                opt.step()
        
        print('Dis sup on edge loss : {}'.format(loss_log.item()))

        log_info = {'loss_head_disen': loss_log.item()}

        return log_info

    def inference(self, data, sparse_edge_index=None):#predict attributes for ind
        #sparse_Edge_index: used in the sparse setting, index of edges require computing attention

        feature, adj = data
        
        fusers = [self.fuse1, self.fuse2]
        if adj.is_sparse:
            adjs = self.models[0].predict_adjs_sparse(feature, adj, fusers, auxiliary_edges=sparse_edge_index)
        else:
            adjs = self.models[0].get_adjs(feature, adj, fusers)

        return adjs


#obtain supervision for capturing all head information
class SupEdgeTrainer(Trainer): 
    #use training on the whole
    def __init__(self, args, model, weight):
        super().__init__(args, model, weight)
        assert args.model == 'DISGAT', "supervision on edges can only be conducted on DISGAT model"

        self.sparse = args.sparse
        self.constrain_layer = args.constrain_layer#0 for all, 1 for the first, 2 for the second

    def get_label_all(self, feature, adj):#anchor_id&adj: tensor; sampled_idx: numpy array
        #return: list of torch tensors
        #check if generated
        if adj.is_sparse:
            adj_tgt = adj.to_dense()
        else:
            adj_tgt = torch.clone(adj)

        adj_tgt = (adj_tgt!=0).float()
        ##remove diagonal elements
        #ind = np.diag_indices(adj_tgt.shape[0])
        #adj_tgt[ind[0], ind[1]] = adj_tgt.new(adj_tgt.shape[0]).fill_(0)

        return adj_tgt.float()


    def sample_train(self, adj):
        # only in the sparse case, sample a batch of edge indices and corresponding values


        label = (adj!=0).float()

        mask = label.new(label.shape).fill_(0)
        #randomly select edge entries for training
        edge_ratio = adj.sum()/(adj.shape[0]*adj.shape[0])
        random_ini_mask = (torch.rand(size=mask.shape) < edge_ratio.item()*3).int()
        mask[:] = random_ini_mask[:]

        #select positive edge entries
        indices = adj.nonzero().cpu().numpy()
        edge_num = int(indices.shape[0]//3)
        np.random.shuffle(indices)
        pos_indices = indices[:edge_num]
        mask[pos_indices[:,0],pos_indices[:,1]] = 1

        #get indeices for sparse inference
        indices = mask.nonzero().transpose(0,1) #(2,edge_num)
        label_used = label[indices[0],indices[1]]


        return label_used, [indices]

    def train_step(self, data, gt_adj=None):#gt_adj is the groundtruth
        #predict dense edges, and supervise them
        #the edges are unnormalized and unfiltered
        
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        loss = None

        if not self.sparse:
            pred_adjs = self.inference(data) #[[A11, A12, ...], [A21, A22, ...]] of size (layer*nhead*node*node). Contains dense edge
            labels = gt_adj
        else:
            labels, indices = self.sample_train(gt_adj)
            pred_adjs = self.inference(data, indices)


        for i, pred_adj in enumerate(pred_adjs):
            if self.constrain_layer==0 or self.constrain_layer==i:
                if self.sparse:
                    adj_agg = []
                    for head in pred_adj:
                        adj_agg.append(head[0])

                    pred = torch.sigmoid(torch.sum(torch.stack(adj_agg), dim=0))

                    if loss is None:
                        loss = utils.adj_mse_loss(pred.squeeze(), labels)
                    else:
                        loss = loss +  utils.adj_mse_loss(pred.squeeze(), labels)

                else:
                    pred_all = torch.sigmoid(torch.sum(torch.stack(pred_adj), dim=0))


                    if loss is None:
                        loss = utils.adj_mse_loss(pred_all, labels)
                    else:
                        loss = loss +  utils.adj_mse_loss(pred_all, labels)

        loss_log = loss
        loss = loss * self.loss_weight
        loss.backward()

        for opt in self.models_opt:
            if self.loss_weight != 0:
                opt.step()
        
        
        print('Sup on heads loss : {}'.format(loss_log.item()))

        log_info = {'loss_heads_sup': loss_log.item()}

        return log_info

    def inference(self, data, sparse_edge_index=None):#predict attributes for ind
        #sparse_Edge_index: used in the sparse setting, index of edges require computing attention

        feature, adj = data
        
        fusers = [self.fuse1, self.fuse2]
        if adj.is_sparse:
            adjs = self.models[0].predict_adjs_sparse(feature, adj, fusers, auxiliary_edges=sparse_edge_index)
        else:
            adjs = self.models[0].get_adjs(feature, adj, fusers)

        return adjs


#Guarantee divergence across heads
class DifHeadTrainer(Trainer): 
    def __init__(self, args, model, weight):
        super().__init__(args, model, weight)
        assert args.model == 'DISGAT', "divergence on heads is only supported for DISGAT"
        
        self.classifier1 = models.MLP(in_feat=args.nhid+args.size, hidden_size=args.nhid, out_size=args.nhead, layers=args.cls_layer)
        self.classifier2 = models.MLP(in_feat=args.nhid*2, hidden_size=args.nhid, out_size=args.nhead, layers=args.cls_layer)
        if args.cuda:
            self.classifier1.cuda()
            self.classifier2.cuda()
        #self.classifier_opt = optim.Adam(list(self.classifier1.parameters())+list(self.classifier2.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        self.classifier_opt1 = optim.Adam(self.classifier1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.classifier_opt2 = optim.Adam(self.classifier2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
        self.models.append(self.classifier1)
        self.models.append(self.classifier2)
        self.models_opt.append(self.classifier_opt1)
        self.models_opt.append(self.classifier_opt2)

        self.nhead = args.nhead
        self.criterion = nn.NLLLoss()

    def get_label_all(self, feature, adj):#anchor_id&adj: tensor; sampled_idx: numpy array
        #not needed


        return None



    def train_step(self, data, pre_dif=None):#pre_dif is not used
        for i, model in enumerate(self.models):
            model.train()
            self.models_opt[i].zero_grad()

        loss = None
        edge_embeds = self.inference(data)


        for layer, edge_embed in enumerate(edge_embeds): #layer
            if layer == 0:
                classifier = self.classifier1
            else:
                classifier = self.classifier2
            for i,edge in enumerate(edge_embed): #nhead
                label = edge.new(edge.shape[0]).fill_(i).long()

                pred_label = classifier(edge,cls=True)

                if loss is None:
                    loss = self.criterion(pred_label, label)
                else:
                    loss = loss + self.criterion(pred_label, label)

        loss_log = loss
        loss = loss * self.loss_weight
        loss.backward()

        for opt in self.models_opt:
            if self.loss_weight != 0:
                opt.step()
        
        
        print('diversity on heads loss : {}'.format(loss_log.item()))

        log_info = {'loss_head_diversity': loss_log.item()}

        return log_info

    def inference(self, data):#predict heads for edge
        #edge_embeds: nlayer list of edge embeddings, of shape(nhead, batchsize, hidden_embedding*2)

        feature, adj = data

        fusers = [self.fuse1, self.fuse2]

        edge_embs = self.models[0].get_edge_em(feature, adj, fusers)

        return edge_embs