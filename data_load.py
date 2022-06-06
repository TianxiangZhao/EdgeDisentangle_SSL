import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import utils
from collections import defaultdict
import os


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    mx = mx.astype(float)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(args, path="data/dblp/", dataset="dblp", edge_type=3):#modified from code: pygcn
    """Load citation network dataset (cora only for now)"""
    #input: idx_features_labels, adj
    #idx,labels are not required to be processed in advance
    #adj: save in the form of edges. idx1 idx2 
    #output: adj, features, labels are all torch.tensor, in the dense form
    #-------------------------------------------------------

    print('Loading {} dataset...'.format(dataset))
    labels = np.load(path+'label.npy')
    if args.origin_feat:
        features = np.load(path+'feature.npy')
    else:
        features = np.load(path+'feature_new.npy')
        
        features = normalize(features)

    edges = []
    for i in range(edge_type):  
        if os.path.exists(path+'adj_{}.npy'.format(i+1)):
            edge = np.load(path+'adj_{}.npy'.format(i+1))
        else:
            edge = sp.load_npz(path+'adj_{}_sp.npz'.format(i+1))
            edge = edge.todense()
        if edge.shape[1] == 2 and edge.shape[0] !=2:#edge is an edge list:
            #change it to an ajacency matrix
            edge = edge.astype(int)
            edge = utils.edge2adj(edge)
        edges.append(edge)

    if args.hetero:
        edges_use = edges
    else:
        edges_use = []
        if args.used_edge == 0:
            adj_new = np.zeros(edges[0].shape)
            for i in range(edge_type):
                adj_new = adj_new + edges[i]
            adj_new = np.clip(adj_new, 0.0, 1.0)

        else:
            adj_new = edges[args.used_edge-1]
        edges_use.append(adj_new)
        
    # process adj
    processed_edges = []
    for adj in edges_use:
        np.fill_diagonal(adj,1)

        adj = adj + np.multiply(adj.T, adj.T > adj) - np.multiply(adj, adj.T > adj)
        
        adj = normalize_adj(adj)

        if args.sparse:
            adj = sp.csr_matrix(adj)
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            processed_edges.append(adj)
        else:
            adj = torch.FloatTensor(np.array(adj))
            processed_edges.append(adj)

    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)

    print('Data loaded')

    if args.hetero:
        #if args.sparse:
         #   raise NotImplementedError('hetero not implemeted for sparse graph')
        #processed_edges = torch.stack(processed_edges)
        return processed_edges, features, labels
    else:
        return processed_edges[0], features, labels


def Extract_graph(edgelist, fake_node, node_num):
    
    node_list = range(node_num+1)[1:]
    node_set = set(node_list)
    adj_1 = sp.coo_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])), shape=(edgelist.max()+1, edgelist.max()+1), dtype=np.float32)
    adj_1 = adj_1 + adj_1.T.multiply(adj_1.T > adj_1) - adj_1.multiply(adj_1.T > adj_1)
    adj_csr = adj_1.tocsr()
    for i in np.arange(node_num):
        for j in adj_csr[i].nonzero()[1]:
            node_set.add(j)

    node_set_2 = node_set
    '''
    node_set_2 = set(node_list)
    for i in node_set:
        for j in adj_csr[i].nonzero()[1]:
            node_set_2.add(j)
    '''
    node_list = np.array(list(node_set_2))
    node_list = np.sort(node_list)
    adj_new = adj_csr[node_list,:]

    node_mapping = dict(zip(node_list, range(0, len(node_list), 1)))

    edge_list = []
    for i in range(len(node_list)):
        for j in adj_new[i].nonzero()[1]:
            if j in node_list:
                edge_list.append([i, node_mapping[j]])

    edge_list = np.array(edge_list)
    #adj_coo_new = sp.coo_matrix((np.ones(len(edge_list)), (edge_list[:,0], edge_list[:,1])), shape=(len(node_list), len(node_list)), dtype=np.float32)

    label_new = np.array(list(map(lambda x: 1 if x in fake_node else 0, node_list)))
    np.savetxt('data/twitter/sub_twitter_edges', edge_list,fmt='%d')
    np.savetxt('data/twitter/sub_twitter_labels', label_new,fmt='%d')

    return


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def norm_sparse(adj):#normalize a torch dense tensor for GCN, and change it into sparse.
    adj = adj + torch.eye(adj.shape[0]).to(adj)
    rowsum = torch.sum(adj,1)
    r_inv = 1/rowsum
    r_inv[torch.isinf(r_inv)] = 0.
    new_adj = torch.mul(r_inv.reshape(-1,1), adj)

    indices = torch.nonzero(new_adj).t()
    values = new_adj[indices[0], indices[1]] # modify this based on dimensionality

    return torch.sparse.FloatTensor(indices, values, new_adj.size())

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def find_shown_index(adj, center_ind, steps = 2):
    seen_nodes = {}
    shown_index = []

    if isinstance(center_ind, int):
        center_ind = [center_ind]

    for center in center_ind:
        shown_index.append(center)
        if center not in seen_nodes:
            seen_nodes[center] = 1

    start_point = center_ind
    for step in range(steps):
        new_start_point = []
        candid_point = set(adj[start_point,:].reshape(-1, adj.shape[1]).nonzero()[:,1])
        for i, c_p in enumerate(candid_point):
            if c_p.item() in seen_nodes:
                pass
            else:
                seen_nodes[c_p.item()] = 1
                shown_index.append(c_p.item())
                new_start_point.append(c_p)
        start_point = new_start_point

    return shown_index

