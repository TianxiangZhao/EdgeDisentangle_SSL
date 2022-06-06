import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import models
import utils
import data_load
import random
import ipdb
import copy
import trainer
import pretrainer
from tensorboardX import SummaryWriter
import os 
import dataset

#from torch.utils.tensorboard import SummaryWriter

# Training setting
parser = utils.get_parser()
args = parser.parse_args()

args.log=True
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.pretrain is not None or args.hnn:
    args.hetero = True
if args.model == 'sage':
    args.sparse=False 
if args.dataset=='dblp':
    args.used_edge=2
    if args.pre_edge is not None:
        for i in range(len(args.pre_edge)):
            args.pre_edge[i] = 2
if args.dataset=='imdb':
    args.used_edge=2
    if args.pre_edge is not None:
        for i in range(len(args.pre_edge)):
            args.pre_edge[i] = 2

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.pre_weight is not None:
    weight_seq = [str(i) for i in args.pre_weight]
else:
    weight_seq = []
    args.pretrain=[]

if args.case:
    args.save = './checkpoint_case/{}/{}_edge{}_reg{}_att{}_gnn{}/task{}/pretrain{}_preweight{}_steps{}_fuse{}_lr{}residue{}relu{}restype{}conLayers{}/'.format(args.dataset, args.model, args.used_edge, args.reg, args.att, args.gnn_type,'_'.join(args.downstream), '_'.join(args.pretrain), '_'.join(weight_seq), args.steps, args.fuse,args.lr, args.residue, args.fuse_no_relu,args.residue_type,args.constrain_layer)
else:
    args.save = './checkpoint/{}/{}_edge{}_reg{}_att{}_gnn{}/task{}/pretrain{}_preweight{}_steps{}_fuse{}_lr{}residue{}relu{}restype{}conLayers{}/'.format(args.dataset, args.model, args.used_edge, args.reg, args.att, args.gnn_type,'_'.join(args.downstream), '_'.join(args.pretrain), '_'.join(weight_seq), args.steps, args.fuse,args.lr, args.residue, args.fuse_no_relu,args.residue_type,args.constrain_layer)

if not os.path.exists(args.save):
    os.makedirs(args.save)

if args.log:
    writer = SummaryWriter(args.save)

#writer = SummaryWriter('SSL')

# Load data
if args.dataset == 'dblp' or args.dataset=='dblp_split':
    adjs, features, labels = data_load.load_data(args)
elif args.dataset == 'imdb':
    adjs, features, labels = data_load.load_data(args,path="data/imdb/", dataset="imdb")
    args.size = 128
elif args.dataset == 'chameleon':
    args.edge_num=1
    adjs, features, labels = data_load.load_data(args,path="data/chameleon/", dataset="chameleon", edge_type=args.edge_num)
    args.size = features.shape[1]
    print('feature dimension: {}'.format(args.size))
elif args.dataset == 'squirrel':
    args.edge_num=1
    adjs, features, labels = data_load.load_data(args,path="data/squirrel/", dataset="squirrel", edge_type=args.edge_num)
    args.size = features.shape[1]
    print('feature dimension: {}'.format(args.size))
elif args.dataset == 'cora_full':
    args.edge_num=1
    adjs, features, labels = data_load.load_data(args,path="data/cora_full/", dataset="cora_full", edge_type=args.edge_num)
    args.size = features.shape[1]
    print('feature dimension: {}'.format(args.size))
elif args.dataset == 'deezer':
    args.edge_num=1
    adjs, features, labels = data_load.load_data(args,path="data/deezer/", dataset="deezer", edge_type=args.edge_num)
    args.size = features.shape[1]
    print('feature dimension: {}'.format(args.size))
elif args.dataset == 'arxiv':
    args.edge_num=1
    adjs, features, labels = data_load.load_data(args,path="data/arxiv/", dataset="arxiv", edge_type=args.edge_num)
    args.size = features.shape[1]
    print('feature dimension: {}'.format(args.size))
elif args.dataset == 'BlogCatalog':
    args.edge_num=1
    adjs, features, labels = data_load.load_data(args,path="data/BlogCatalog/", dataset="BlogCatalog", edge_type=args.edge_num)
    args.size = features.shape[1]
    print('feature dimension: {}'.format(args.size))
elif args.dataset == 'cora':
    args.edge_num=1
    adjs, features, labels = data_load.load_data(args,path="data/cora/", dataset="cora", edge_type=args.edge_num)
    args.size = features.shape[1]
    print('feature dimension: {}'.format(args.size))
else:
    print("no this dataset: {args.dataset}")

args.nclass = labels.max().item() + 1


print(args)

# Model and optimizer
if not args.hnn: #not using heterogeneous GNN
    if args.model == 'sage':
        encoder = models.GraphSage(args,nfeat=args.size, 
                nhid=args.nhid, 
                nclass=args.nhid, 
                dropout=args.dropout)
    elif args.model == 'sage2':
        encoder = models.Sage2(args,nfeat=args.size, 
                nhid=args.nhid, 
                nclass=args.nhid, 
                dropout=args.dropout)
    elif args.model == 'gcn':
        encoder = models.GCN(args, nfeat=args.size, 
                nhid=args.nhid, 
                nclass=args.nhid, 
                dropout=args.dropout)
    elif args.model == 'GAT':
        encoder = models.GAT(args, nfeat=args.size, 
                nhid=args.nhid, 
                nclass=args.nhid,
                nheads=args.nhead, 
                dropout=args.dropout)
    
    elif args.model == 'DISGAT':
        encoder = models.DISGAT(args, nfeat=args.size, 
                nhid=args.nhid, 
                nclass=args.nhid,
                nheads=args.nhead, 
                dropout=args.dropout)

    elif args.model == 'GIN':
        encoder = models.GIN(input_dim=args.size, 
                hidden_dim=args.nhid, 
                output_dim=args.nhid)
    
    elif args.model == 'FactorGCN':
        encoder = models.FactorGCN(args, nfeat=args.size, 
                nhid=args.nhid, 
                nclass=args.nhid,
                nheads=args.nhead)

    elif args.model == 'MLP':
        encoder = models.MLPEncoder(args, nfeat=args.size, 
                nhid=args.nhid, 
                nclass=args.nhid, 
                dropout=args.dropout)

    elif args.model == 'Mixhop':
        args.nhid = 64
        encoder = models.MixHop(args.size, args.nhid//2, args.nhid//2, num_layers=2,
                       dropout=args.dropout, hops=1)
    
    elif args.model == 'H2GCN':
        if args.hetero:
            adj = adjs[args.used_edge-1]
        else:
            adj = adjs
        encoder = models.H2GCN(args.size, args.nhid, args.nhid, adj,
                        features.shape[0],
                        num_layers=2, dropout=args.dropout,
                        num_mlp_layers=1)
        

    

else:
    adj = adjs
    if args.model == 'RGCN':
        classifier = models.RelationalGraphConvModel(input_size=args.size, 
                hidden_size=args.nhid, 
                output_size=labels.max().item() + 1,
                num_rel=args.edge_num, 
                dropout=args.dropout)
    if args.model == 'HAN':
        num_head_list = [4,4]
        classifier = models.HAN(in_size=args.size, 
                hidden_size=args.nhid, 
                out_size=labels.max().item() + 1, 
                num_heads=num_head_list,
                num_meta_paths=args.edge_num,
                dropout=args.dropout)


if args.cuda:
    encoder = encoder.cuda()
    features = features.cuda()
    if args.hetero:#if heterogeneous graph:
        for i in range(len(adjs)):
            adjs[i] = adjs[i].cuda()
    else:
        adjs = adjs.cuda()#if adj is a list, how can it be put to cuda?
    labels = labels.cuda()



def save_model(epoch):
    saved_content = {}
    saved_content['encoder'] = encoder.state_dict()

    path = './checkpoint/{}/{}_used_edge{}_weight{}_reg{}'.format(args.dataset, args.model, args.used_edge, args.pre_weight, args.reg)
    if not os.path.exists(path):
        os.makedirs(path)

    #torch.save(saved_content, 'checkpoint/{}/{}_epoch{}_edge{}_{}.pth'.format(args.dataset,args.model,epoch, args.used_edge, args.method))
    torch.save(saved_content, path+'/pretrain_{}_{}.pth'.format(args.pretrain, epoch))

    
    print("successfully saved: {}".format(epoch))
    return

def load_model():
    loaded_content = torch.load('./checkpoint/{}/{}_used_edge{}_weight{}_reg{}/pretrain_{}_{}.pth'.format(args.dataset, args.model, args.used_edge, args.pre_weight, args.reg, args.pretrain, args.load), map_location=lambda storage, loc: storage)

    encoder.load_state_dict(loaded_content['encoder'])

    print("successfully loaded: {}".format(args.load))
    return

SSL_Trainer_dict={'PredAttr': pretrainer.PredTrainer,'PredDistance': pretrainer.DistanceTrainer, 'PredContext':pretrainer.ContextPredTrainer, 'DisEdge': pretrainer.GeneratedEdgeTrainer,'SupEdge': pretrainer.SupEdgeTrainer,'DifHead': pretrainer.DifHeadTrainer}

#initialize pretrainer
ssl_trainers=[]
ssl_labels=[]
if args.pretrain is not None:
    for i, ssl_signal in enumerate(args.pretrain):
        assert args.pre_edge[i]>0, "edge index begins from 1"
        SSLtrainer = SSL_Trainer_dict[ssl_signal](args, encoder, args.pre_weight[i])
        if ssl_signal !='DisEdge':
            SSLabel = SSLtrainer.get_label_all(features, adjs[args.used_edge-1])
        else:
            SSLabel = SSLtrainer.get_label_all(features, adjs[args.used_edge-1], labels)
        ssl_trainers.append(SSLtrainer)
        ssl_labels.append(SSLabel)

DOWNtrainers=[]
Downstream_Trainer_dict={'CLS': trainer.ClsTrainer, 'Edge': trainer.EdgeTrainer}
if args.downstream is not None:
    for i, signal in enumerate(args.downstream):
        DOWNtrainer = Downstream_Trainer_dict[signal](args, encoder, labels, args.down_weight[0])#labels are used for train/valid/test split
        DOWNtrainers.append(DOWNtrainer)


# Train model
if args.load is not None:
    load_model()

if args.batch:
    graphset = dataset.LoadProcessedDataset(args)

t_total = time.time()
steps = args.steps
for epoch in range(args.epochs):
    if epoch % 40 == 0:
        adj = adjs
        if not args.hnn: #
            if args.hetero:
                adj = adjs[args.used_edge-1]
        for i, trainer in enumerate(DOWNtrainers):

            if not args.batch:
                inputs = [features, adj]        
                label = labels
                log_info = trainer.test(inputs, label, epoch)
            else:
                log_info = trainer.test_batch(graphset, labels, True)

            for key in log_info:
                if args.log:
                    writer.add_scalar(args.downstream[i]+key, log_info[key], epoch*steps) 
            #writer.add_scalars(args.downstream[i]+'test', log_info, epoch)

        #case study on disentanglement
        if args.case:

            with torch.no_grad():
                at_distance_list, correlation_graph_lists, feat_dim_graph_lists = trainer.analyze_disentangle(features, adj)

            #visualize correlation of feature dimensions
            writer.add_scalar(args.downstream[i]+'att_correlation_layer1', at_distance_list[0], epoch*steps)
            writer.add_scalar(args.downstream[i]+'att_correlation_layer2', at_distance_list[1], epoch*steps)

            for layer in range(2):
                writer.add_figure('correlation_coeff on attention head at layer {}'.format(layer), utils.draw_heatmap(correlation_graph_lists[layer]),epoch*steps)
                writer.add_figure('correlation_coeff on feature dim at layer {}'.format(layer), utils.draw_heatmap(feat_dim_graph_lists[layer]),epoch*steps)

    if args.batch:
        if args.finetune:
            batch_num = int((graphset.tot_num*args.node_sup_ratio)//args.batch_size)
        else:
            batch_num = graphset.tot_num//args.batch_size
    else:
        batch_num = 1
    for batch in range(batch_num):
        #ipdb.set_trace()
        if args.finetune:

            for step in range(steps):
                adj = adjs
                if not args.hnn: #
                    if args.hetero:
                        adj = adjs[args.used_edge-1]

                for i, trainer in enumerate(DOWNtrainers):
                    if not args.batch:
                        inputs = [features, adj]        
                        label = labels
                        log_info = trainer.train_step(inputs, label, epoch)
                    else:
                        log_info = trainer.train_batch(graphset, labels, batch)

                    for key in log_info:
                        if args.log:
                            writer.add_scalar(args.downstream[i]+key, log_info[key], epoch*batch_num*steps+batch+step)
                
                    #writer.add_scalars(args.downstream[i], log_info, epoch)

        if args.pretrain is not None:
            for i, trainer in enumerate(ssl_trainers):
                if not args.batch:
                    if args.hetero and not args.hnn:
                        adj = adjs[args.pre_edge[i]-1]
                    else:
                        adj=adjs
                    inputs = [features, adj]
                    pre_label = ssl_labels[i]
                    log_info = trainer.train_step(inputs, pre_label)
                else:
                    pre_label = ssl_labels[i]
                    log_info = trainer.train_batch(graphset, pre_label, batch)

                #writer.add_scalars(args.pretrain[i], log_info, epoch)
                for key in log_info:
                    if args.log:
                        writer.add_scalar(args.pretrain[i]+key, log_info[key], (epoch*batch_num+batch)*steps)

    if epoch % 100 == 0:
        if args.downstream is None:
            #save_model(epoch)
            print('no save for now')

        else:
            print('no save for now')
if args.log:
    writer.close()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

