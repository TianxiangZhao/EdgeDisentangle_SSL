import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform
from scipy.sparse import coo_matrix
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--sparse', action='store_true', default=False,
                    help='whether use sparse adj matrix')
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--nhid', type=int, default=64)#intermediate feature dimension
    parser.add_argument('--nclass', type=int, default=5)#number of labels
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument('--size', type=int, default=64) # input feature dimension
    parser.add_argument('--epochs', type=int, default=510,
                help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_nums', type=int, default=6000, help='number of batches per epoch')

    parser.add_argument('--load', type=int, default=None) #load from pretrained model under the same setting, indicate load epoch
    parser.add_argument('--save', type=str, default=None)

    parser.add_argument('--log', action='store_true', default=False,
                    help='whether save logs and checkpoints')

    parser.add_argument('--method', type=str, default='no', 
        choices=['no',])
    parser.add_argument('--model', type=str, default='DISGAT', 
        choices=['sage','gcn','GAT','sage2','MLP','RGCN','HAN', 'DISGAT', 'GIN', 'FactorGCN', 'Mixhop', 'H2GCN'])
    parser.add_argument('--nhead', type=int, default=4)#intermediate feature dimension
    parser.add_argument('--hetero', action='store_true', default=False,
                    help='whether using multiple edge types.')
    parser.add_argument('--hnn', action='store_true', default=False,
                    help='whether use heterogeneous GNN.')
    parser.add_argument('--edge_num', type=int, default=3,
                    help='number of edge types')
    parser.add_argument('--used_edge', type=int, default=1,
                    help='0: using all egde. 1, 2, 3...: use only corresponding edge type.')
    parser.add_argument('--cls_layer', type=int, default=2,
                    help='number of layers in classifier. Must be larger than 0')
    parser.add_argument('--EdgePred_layer', type=int, default=1,
                    help='number of layers in classifier. Must be larger than 0')

    parser.add_argument("--downstream",nargs='+', type=str, choices=['CLS','Edge'])#choices=['']
    parser.add_argument("--down_weight",nargs='+', type=float)
    parser.add_argument("--pretrain",nargs='+', type=str, choices=['PredAttr','PredDistance', 'PredContext', 'DisEdge','SupEdge', 'DifHead'])#choices=['']
    parser.add_argument("--pre_weight",nargs='+', type=float)
    parser.add_argument("--pre_edge",nargs='+', type=int)

    parser.add_argument('--finetune', action='store_true', default=False,
                    help='whether to train towards target task')
    parser.add_argument('--enc_layer', type=int, default=2, help='number of layers in the encoder')
    parser.add_argument('--fuse', type=str, default='last', 
        choices=['last','avg','concat'])
    parser.add_argument('--pretext_dim', type=int, default=16, help='dimension for pre-computed context in context-prediction task')
    parser.add_argument('--cluster_num', type=int, default=16, help='clusters for pre-computed context')
    parser.add_argument('--node_sup_ratio', type=float, default=0.25, help='ratio of nodes labeled')

    
    parser.add_argument('--reg', action='store_true', default=False,
                    help='whether to regularize weight in fusers')
    parser.add_argument('--reg_weight', type=float, default=0.01, help='weight of l1 norm on fusers')
    
    parser.add_argument('--batch', action='store_true', default=False,
                    help='whether use batches of sub-graphs as data')
    parser.add_argument('--batch_size', type=int, default=40, help='number of batches per epoch')
    parser.add_argument('--SubgraphSize', type=int, default=128, help='size of subgraphs to be used')
    parser.add_argument('--origin_feat', action='store_true', default=False,
                    help='whether to use original feature')

    parser.add_argument('--att', type=int, default=2, help='Type of attention used. 1 for product with prototype, otherwise inner-product')
    parser.add_argument("--dis_type", type=int, default=1, help='Type of supervision on disentangled edges. 1 for homo, 2 for class-homo')
    parser.add_argument("--constrain_layer", type=int, default=0, help='Layer of constrained edge. 0 for all, 1 for the first, 2 for the second')
    
    parser.add_argument('--residue', action='store_true', default=False,
                    help='whether use residue for DISGAT model')    
    parser.add_argument('--fuse_no_relu', action='store_true', default=False,
                    help='whether use relu in fuser layer')
    parser.add_argument('--residue_type', type=int, default=0,
                    help='0: out = 1-MLP(nhead*head_emb+residue),1: out = 2-MLP(nhead*head_emb+residue),2: out = 1-MLP(nhead*head_emb)+1-MLP(residue),')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--gnn_type', type=str, default='AT', choices=['AT', 'SAGE', 'GCN'], help='type of GNN in DISGAT')
    
    parser.add_argument('--case', action='store_true', default=False,
                    help='whether case study mode')    
    
    parser.add_argument('--conformT', action='store_true', default=False,
                    help='whether case study on label conformity loss')    
    return parser

def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)
    return

def split(labels,train_ratio=0.25):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    val_ratio = (1-train_ratio)/4
    test_ratio = (1-train_ratio)/4*3

    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <11:
            print("too small class type: {}, num{}".format(i, c_num))
            ipdb.set_trace()
        else:
            c_num_mat[i,0] = int(c_num*train_ratio)
            c_num_mat[i,1] = int(c_num*val_ratio)
            c_num_mat[i,2] = int(c_num*test_ratio)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat

def edge2adj(edgelist):
    edges = np.array(edgelist)
    #ipdb.set_trace()
    adj = np.zeros((int(edgelist.max())+1,int(edgelist.max())+1))
    for edge in edges:
        adj[edge[0]][edge[1]] = 1

    return adj

def tensor2coo(x):
        """ converts tensor x to scipy coo matrix """

        node_num = x.shape[0]
        if not x.is_sparse:
            indices = torch.nonzero(x)
            indices = indices.t()
            values = x[list(indices[i] for i in range(indices.shape[0]))].cpu().numpy()
        else:
            indices = x.coalesce().indices()  
            values = x.coalesce().values().cpu().numpy()
        if len(indices.shape) == 0:  # if all elements are zeros
            return coo_matrix((node_num, node_num), dtype=np.int8)
        
        row = indices[0,:].cpu().numpy()
        column = indices[1,:].cpu().numpy()
        

        return coo_matrix((values,(row,column)), shape=(node_num, node_num))     

def sp_softmax(indices, values, N):
    source, _ = indices
    v_max = values.max()
    exp_v = torch.exp(values - v_max)
    exp_sum = exp_v.new(N, 1).fill_(0)
    exp_sum.scatter_add_(0, source.unsqueeze(1), exp_v)
    exp_sum += 1e-10
    softmax_v = exp_v / exp_sum[source]
    return softmax_v


def sp_matmul(indices, values, mat):
    source, target = indices
    out = torch.zeros_like(mat)
    out.scatter_add_(0, source.expand(mat.size(1), -1).t(), values * mat[target])
    return out

def print_edges_num(dense_adj, labels):
    c_num = labels.max().item()+1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)

    for i in range(c_num):
        for j in range(c_num):
            #ipdb.set_trace()
            row_ind = labels == i
            col_ind = labels == j

            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            print("edges between class {:d} and class {:d}: {:f}".format(i,j,edge_num))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def print_class_acc(output, labels, class_num_list, pre='valid'):
    pre_num = 0
    #print class-wise performance
    '''
    for i in range(labels.max()+1):
        
        cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(output, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(output, dim=-1)[:,1].detach().cpu(), average='macro')

    macro_F = f1_score(labels.detach().cpu(), torch.argmax(output, dim=-1).detach().cpu(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    return

def Roc_F(output, labels, class_num_list, pre='valid'):
    pre_num = 0
    #print class-wise performance
    '''
    for i in range(labels.max()+1):
        
        cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(output, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(output, dim=-1)[:,1].detach().cpu(), average='macro')

    macro_F = f1_score(labels.detach().cpu(), torch.argmax(output, dim=-1).detach().cpu(), average='macro')
    #print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    return auc_score, macro_F


def adj_mse_loss(adj_rec, adj_tgt):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0]**2

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.mean(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss

def masked_adj_mse_loss(adj_rec, adj_tgt, mask):
    edge_num = (adj_tgt*mask).nonzero().shape[0]
    total_num = mask.nonzero().shape[0]

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight
    weight_matrix[mask==0] = 0

    loss = torch.mean(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss

def adj_accuracy(adj_rec, adj_tgt):#require that edges are 0 or 1
    total_num = adj_tgt.shape[0]*adj_tgt.shape[1]
    pos_num = (adj_tgt==1).sum()
    neg_num = (adj_tgt==0).sum()
    assert pos_num+neg_num == total_num, 'value of edges is incorrect for calculating accuracy'

    correct_num = adj_rec.eq(adj_tgt).double().sum()
    TP_num = (adj_rec[adj_tgt==1]==1).double().sum()
    TN_num = (adj_rec[adj_tgt==0]==0).double().sum()

    return correct_num/total_num, TP_num/pos_num, TN_num/neg_num

def group_correlation(embedding):
    #embedding: tensor of shape n_node*n_embed
    #return: correlation coefficient map of size (n_node, n_node)
    x = embedding - torch.mean(embedding,dim=-1).unsqueeze(-1)
    x_norm = torch.sqrt(torch.diagonal(torch.mm(x,x.transpose(0,1)),0))

    corre_map = (torch.mm(x,x.transpose(0,1)))/torch.ger(x_norm, x_norm)

    return corre_map

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    #ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def draw_heatmap(matrix):
    if torch.is_tensor(matrix):
        matrix = matrix.cpu().numpy()
    matrix = np.abs(matrix)

    #arrange size of color bar
    plt.rc('axes', labelsize=20)
    
    x_index = np.arange(matrix.shape[0]).tolist()
    y_index = np.arange(matrix.shape[0]).tolist()

    fig, ax = plt.subplots()

    im, cbar = heatmap(matrix, x_index, y_index, ax=ax,
                   cmap="PuOr", vmin=0, vmax=1,
                cbarlabel="correlation coeff.")
    #texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    fig.tight_layout()

    return fig


    






