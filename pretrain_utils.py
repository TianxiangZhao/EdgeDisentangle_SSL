import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import ipdb
from sklearn.decomposition import PCA
import sklearn.cluster as cluster
import networkx as nx

import multiprocessing as mp
from functools import partial
import utils



def obtain_context(features, adj, index, ): #obtain features from neighborhoods and use as context, adj is torch.tensor or torch.sparseTensor
    #inputed adj are already normalized, hence use sum, instead of mean

    if not adj.is_sparse:
        adj = adj
        
        #context_emb = torch.stack([torch.sum(torch.einsum('nm,n->nm', features,adj[ind]), dim=0) for ind in index], dim=0)
        if len(adj.shape)==3:#batch senario
            context_emb = torch.matmul(adj, features)
        else:
            context_emb = torch.matmul(adj[index], features)
    else:
        adj = adj.to_dense()[index, :]
        adj = adj.to_sparse()
        context_emb = torch.matmul(adj, features)

    return context_emb

def obtain_pretext(features, dim):#PCA to get pretext attributes for attribute-prediction SSL
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    pca = PCA(n_components=dim)
    pca.fit(features)

    result = pca.transform(features)

    return result

def context_dist_node(center_node, adj, context): # return a numpy array containing context dist
    if torch.is_tensor(adj):
        adj = utils.tensor2coo(adj)
    if torch.is_tensor(context):
        context = context.cpu().numpy()#.astype(int)

    adj_csr = adj.tocsr()

    center_neighbor = adj_csr.getrow(center_node).nonzero()[0]
    neighbor_context = context[center_neighbor]

    context_set = set(context.tolist())
    context_dict = dict([(key, index) for index, key in enumerate(context_set)])

    context_dist = np.zeros(len(context_set))

    for item in neighbor_context:
        context_dist[context_dict[item]] = context_dist[context_dict[item]]+1

    context_dist = context_dist/(np.sum(context_dist)+0.001)

    return context_dist

def context_dist_all(features, adj, cluster_num): #clusters to get context distribution
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    if torch.is_tensor(adj):
        adj = utils.tensor2coo(adj)
    
    
    features = obtain_pretext(features, 16)

    #filter outliers
    kmeans = cluster.KMeans(n_clusters=100, random_state=0).fit(features)
    clust_label = kmeans.labels_
    unique, counts = np.unique(clust_label, return_counts=True)
    selected_clusts = unique[counts>=20]
    selected_index = []
    for clust in selected_clusts:
        selected_index+=(np.where(clust_label==clust)[0].tolist())
        

    kmeans = cluster.KMeans(n_clusters=cluster_num, random_state=0).fit(features[selected_index,:])
    clust_label = kmeans.predict(features)

    
    unique, counts = np.unique(clust_label, return_counts=True)
    print('instances belong to each cluster: {}'.format(counts))

    node_list = np.arange(features.shape[0]).tolist()
    with mp.Pool(4) as p:
        result = p.map(partial(context_dist_node, adj=adj, context=clust_label), node_list)
        

    dist_all = np.vstack(result)


    return dist_all

def compute_dist_to_node(node, G):

    dist_dict = dict(nx.single_target_shortest_path_length(G, node))
    for i in list(G.nodes):
        if i not in dist_dict:
            #set 999 as the maximum distance
            dist_dict[i] = 999

    dist_array = np.array([[k,v] for k, v in dist_dict.items()])
    dist = dist_array[dist_array[:,0].argsort(),1]

    return dist


def compute_dist_to_group(adj, anchor_chosen):
    """
    return a normalized dist array, node_num*anchor_num
    """
    assert anchor_chosen.shape[0]%4 ==0, "anchor number should be proportional to 4(number of workers)"
    #ipdb.set_trace()
    #change to a networkx object
    if torch.is_tensor(adj):
        adj = utils.tensor2coo(adj)

    G = nx.Graph()
    G.add_nodes_from(np.arange(adj.shape[0]))

    coo_adj = adj.tocoo()
    edge_list = [(coo_adj.row[i], coo_adj.col[i]) for i in np.arange(coo_adj.nnz)]
    G.add_edges_from(edge_list)

    if not isinstance(anchor_chosen, list):
        anchor_chosen = anchor_chosen.tolist()

    #count the pair-wise distance towards the chosen anchors
    with mp.Pool(4) as p:
        result = p.map(partial(compute_dist_to_node, G=G), anchor_chosen)

    dist_embed = np.vstack(result).T  #[all, anchor]

    #dist_embed = 1.0/(1+dist_embed)


    return dist_embed

def compute_dist_all(adj):
    '''
    return pairwise distance matrix
    '''
    
    if torch.is_tensor(adj):
        adj = utils.tensor2coo(adj)

    G = nx.Graph()
    G.add_nodes_from(np.arange(adj.shape[0]))

    coo_adj = adj.tocoo()
    edge_list = [(coo_adj.row[i], coo_adj.col[i]) for i in np.arange(coo_adj.nnz)]
    G.add_edges_from(edge_list)

    dist_matrix = np.zeros(adj.shape)
    p = dict(nx.shortest_path_length(G))
    for src in range(adj.shape[0]):
        for tgt in range(src):
            if tgt in p[src].keys():
                dist_matrix[src][tgt] = p[src][tgt]

    dist_matrix = dist_matrix + dist_matrix.transpose()
    dist_matrix[dist_matrix==0] = 99
    np.fill_diagonal(dist_matrix, 0)

    return dist_matrix

