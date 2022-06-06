import numpy as np 
import utils
import ipdb
import scipy.sparse as sp
import random
import time
import copy
import torch
import torch.nn.functional as F
import pickle
from data_load import load_data
import argparse
import os


def uniform_sub_graph(adj, chosen_ind, batch_node_num, node_attr=None, require_valid_ind=False):
    #used to process all sub-graphs in a batch to the same shape
    #require_valid_ind: whether require the indices of all used nodes
    ###output:
    ###    #last_valid_ind: number of valid nodes in current sub-graph

    cur_node_num = chosen_ind.shape[0]
    add_num = 0

    last_valid_ind = batch_node_num
    if cur_node_num > batch_node_num:
        chosen_ind = np.random.choice(chosen_ind, batch_node_num, replace=False)

    elif cur_node_num < batch_node_num:
        add_num = batch_node_num - cur_node_num
        last_valid_ind = cur_node_num

    adj_extracted = adj[chosen_ind,:].tocsc()[:,chosen_ind]

    valid_adj = adj_extracted#used for getting position embedding
    valid_ind = chosen_ind

    if node_attr is not None:
        node_extracted = node_attr[chosen_ind,:]
        if add_num !=0:
            add_feat = np.zeros((add_num, node_extracted.shape[1]))
            node_extracted = np.concatenate((node_extracted, add_feat), axis=0)
    else:
        node_extracted = None
        
    if add_num != 0:
        adj_extracted = np.zeros((batch_node_num, batch_node_num))
        adj_extracted[:cur_node_num, :cur_node_num] = np.array(valid_adj.todense())
    else:
        adj_extracted = np.array(adj_extracted.todense())

    adj_extracted = np.clip(adj_extracted, 0.0,1.0)


    
    if require_valid_ind:
        return adj_extracted, node_extracted, last_valid_ind, valid_ind
    else:
        return adj_extracted, node_extracted, last_valid_ind

class GraphDataset(object):
    #used for process&return batch of small subgraphs
    def __init__(self, args):
        self.args = args
        self.cur_ind = 0

    def return_full_index(self):

        raise NotImplementedError

    def get_sample(self, index):

        raise NotImplementedError

    def get_random_sample(self, valid_index=None, num=1):

        raise NotImplementedError

    def get_ordered_sample(self, num=1):

        raise NotImplementedError

    def get_whole(self):

        raise NotImplementedError


class RawGraph(GraphDataset):
    """process loaded one large graph"""
    def __init__(self, args, adj, label, feature=None):
        '''
        require adj in: csr matirx
        require feature in: np.array
        '''
        super(RawGraph, self).__init__(args)

        self.adj = adj.cpu().numpy()
        self.adj = sp.csr_matrix(self.adj)
        self.feature = feature.cpu().numpy()
        self.label = label.cpu().numpy()

        self.tot_num = adj.shape[0]

        assert isinstance(self.adj, sp.csr_matrix), "cannot obtain subgraph, wrong data type. expecting csr matrix"


    def return_full_index(self):

        return np.arange(self.adj.shape[0])


    def obtain_sub_graph_ind(self, valid_ind=None, step=2, given_ind = None):
        # given_ind: set the ind of sub_graph to be obtained
        # valid_ind: numpy 1-D array of valid node indices 
        
        adj = self.adj

        if given_ind is not None:
            center_ind = given_ind
        elif valid_ind is None:
            center_ind = np.random.randint(0,adj.shape[0])
        else:
            center_ind = np.random.choice(valid_ind, 1)[0]


        chosen_set = set([center_ind])

        for s in range(step):
            #enlarge_set = set.union(*[set(adj[node].nonzero()[1]) for node in chosen_set])

            #add a neighbor sampler. need to make sure they should be within a range
            enlarge_set = set()
            for node in chosen_set:
                new_set = set(adj[node].nonzero()[1])
                if len(new_set) > 15 and len(new_set) <=30:
                    new_set = set(random.sample(new_set, len(new_set)//2))
                elif len(new_set) > 30:
                    new_set = set(random.sample(new_set, len(new_set)//3))

                enlarge_set = set.union(enlarge_set, new_set)

            chosen_set = set.union(chosen_set, enlarge_set)

        chosen_ind = np.array(list(chosen_set))

        return chosen_ind

    def get_random_sample(self, valid_index=None, num=1):
        #for source graph:
        extracted_ind = []
        extracted_adj = []
        extracted_node = []
        extracted_valid_ind = []
        extracted_label = []

        batch_node_num = 0

        for i in range(num):

            #sample a center node
            chosen_ind = self.obtain_sub_graph_ind(valid_index)

            batch_node_num = batch_node_num + chosen_ind.shape[0]
            extracted_ind.append(chosen_ind)

        batch_node_num = batch_node_num // num


        #adjust the graph size, form a batch
        for i in range(num):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, last_valid_ind, valid_ind = uniform_sub_graph(self.adj, chosen_ind, batch_node_num ,self.feature, require_valid_ind = True)
            extracted_ind[i] = valid_ind

            extracted_adj.append(adj_extracted)
            extracted_node.append(node_extracted)
            extracted_valid_ind.append(last_valid_ind)
            extracted_label.append(self.label[chosen_ind])

        #change into torch tensor
        if self.args.cuda:
            extracted_adj = torch.FloatTensor(extracted_adj).cuda()
            extracted_node = torch.FloatTensor(extracted_node).cuda()
            extracted_valid_ind = torch.IntTensor(extracted_valid_ind).cuda()
            extracted_label = torch.IntTensor(extracted_label).cuda()
        else:
            extracted_adj = torch.FloatTensor(extracted_adj)
            extracted_node = torch.FloatTensor(extracted_node)
            extracted_valid_ind = torch.IntTensor(extracted_valid_ind)
            extracted_label = torch.IntTensor(extracted_label)


        return extracted_adj, extracted_node, extracted_valid_ind, extracted_label

    def get_tosave_batch(self, valid_index=None, num=1, preset_size=None):
        ###preset_size: size of sub-graphs in a batch
        if valid_index is not None:
            tot_num = len(valid_index)
        else:
            tot_num = self.tot_num

        if self.cur_ind+num >= tot_num:
            self.cur_ind = 0
            print("finish obtain one epoch of data")

        #for source graph:
        extracted_ind = []
        extracted_adj = []
        extracted_node = []
        extracted_valid_ind = []
        extracted_label = []

        batch_node_num = 0

        for i in range(num):

            #sample a center node
            chosen_ind = self.obtain_sub_graph_ind(valid_index, given_ind = self.cur_ind+i)

            batch_node_num = batch_node_num + chosen_ind.shape[0]
            extracted_ind.append(chosen_ind)

        batch_node_num = batch_node_num // num

        if preset_size is not None:
            batch_node_num = preset_size

        #adjust the graph size, form a batch
        for i in range(num):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, last_valid_ind, valid_ind = uniform_sub_graph(self.adj, chosen_ind, batch_node_num ,self.feature, require_valid_ind = True)
            extracted_ind[i] = valid_ind

            extracted_adj.append(adj_extracted)
            extracted_node.append(node_extracted)
            extracted_valid_ind.append(last_valid_ind)
            extracted_label.append(self.label[chosen_ind[0]])



        extracted_adj = np.array(extracted_adj)
        extracted_node = np.array(extracted_node)
        extracted_valid_ind = np.array(extracted_valid_ind)
        extracted_label = np.array(extracted_label)

        self.cur_ind = self.cur_ind + num

        return extracted_node, extracted_adj, extracted_valid_ind, extracted_label


    def save_batched_pickle(self, batch_id, node, adj, valid_ind, label):
        ################################
        #used for preprocessing dataset
        #data form: a standard form. When load, need to be processed.
        #node: batchsize*node*embedding_size
        #adj: batchsize*node*node
        ################################



        #save 
        folder = './data/{}_split/'.format(self.args.dataset)+str(adj.shape[1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        for index in range(node.shape[0]):
            data = {}
            data['node'] = node[index]
            data['adj'] = adj[index]
            data['valid_ind'] = valid_ind[index]
            data['label'] = label[index]

            number = batch_id*node.shape[0]+index
            address = folder+'/'+str(number)+'.p'
            print('save '+address)
            pickle.dump(data, open(address, "wb"))


    def load_batched_pickle(self, valid_index=None, num=1, preset_size=None):
        if valid_index is not None:
            tot_num = len(valid_index)
        else:
            tot_num = self.tot_num

        if self.cur_ind+num >= tot_num:
            self.cur_ind = 0
            print("finish obtain one epoch of data")

        #for source graph:
        extracted_ind = []
        extracted_adj = []
        extracted_node = []
        extracted_valid_ind = []
        extracted_label = []

        for i in range(num):

            #sample a center node
            addr = './dataset/{}_split/'.format(self.args.dataset)+str(preset_size)+'/'+str(self.cur_ind+i)+'.p'
            data = pickle.load(open(addr, "rb"))

            

            extracted_adj.append(data['adj'])
            extracted_node.append(data['node'])
            extracted_valid_ind.append(data['valid_ind'])
            extracted_label.append(data['label'])

        extracted_adj = np.array(extracted_adj)
        extracted_node = np.array(extracted_node)
        extracted_valid_ind = np.array(extracted_valid_ind)
        extracted_label = np.array(extracted_label)

        self.cur_ind = self.cur_ind + num


        return extracted_node, extracted_adj, extracted_valid_ind


class LoadProcessedDataset(GraphDataset):
    """docstring for PairedTransGraph"""
    def __init__(self, args):
        '''
        require adj in: csr matirx
        require feature in: np.array
        '''
        super(LoadProcessedDataset, self).__init__(args)
        self.args = args

        if args.dataset == 'dblp_split':
            self.tot_num = 4057
        else:
            raise ValueError("this dataset is not processed yet")

        self.cur_ind = 0

    def return_full_index(self):

        return np.arange(self.tot_num)


    def get_ordered_sample(self, valid_index=None, num=1, preset_size=100):
        if valid_index is not None:
            tot_num = len(valid_index)
        else:
            tot_num = self.tot_num

        if self.cur_ind+num >= tot_num:
            self.cur_ind = 0
            print("finish obtain one epoch of data")

        #for source graph:
        extracted_ind = []
        extracted_adj = []
        extracted_node = []
        extracted_valid_ind = []
        extracted_label = []

        for i in range(num):

            #sample a center node
            addr = './data/{}/'.format(self.args.dataset)+str(preset_size)+'/'+str(valid_index[self.cur_ind+i])+'.p'
            data = pickle.load(open(addr, "rb"))

            

            extracted_adj.append(data['adj'])
            extracted_node.append(data['node'])
            extracted_valid_ind.append(data['valid_ind'])
            extracted_label.append(data['label'])



        extracted_adj = np.array(extracted_adj)
        extracted_node = np.array(extracted_node)
        extracted_valid_ind = np.array(extracted_valid_ind)
        extracted_label = np.array(extracted_label)

        if self.args.cuda:
            extracted_adj = torch.FloatTensor(extracted_adj).cuda()
            extracted_node = torch.FloatTensor(extracted_node).cuda()
            extracted_valid_ind = torch.IntTensor(extracted_valid_ind).cuda()
            extracted_label = torch.IntTensor(extracted_label).cuda()

        else:
            extracted_adj = torch.FloatTensor(extracted_adj)
            extracted_node = torch.FloatTensor(extracted_node)
            extracted_valid_ind = torch.IntTensor(extracted_valid_ind)
            extracted_label = torch.IntTensor(extracted_label)
        
        self.cur_ind = self.cur_ind + num

        return extracted_node, extracted_adj, extracted_valid_ind, extracted_label

        
    def get_batch(self, batch_index, preset_size=100):

        #for source graph:
        extracted_ind = []
        extracted_adj = []
        extracted_node = []
        extracted_valid_ind = []
        if torch.is_tensor(batch_index):
            batch_index = batch_index.cpu().numpy()

        for ind in batch_index:

            #sample a center node
            addr = './data/{}/'.format(self.args.dataset)+str(preset_size)+'/'+str(ind)+'.p'
            data = pickle.load(open(addr, "rb"))

            extracted_adj.append(data['adj'])
            extracted_node.append(data['node'])
            extracted_valid_ind.append(data['valid_ind'])



        extracted_adj = np.array(extracted_adj)
        extracted_node = np.array(extracted_node)
        extracted_valid_ind = np.array(extracted_valid_ind)

        if self.args.cuda:
            extracted_adj = torch.FloatTensor(extracted_adj).cuda()
            extracted_node = torch.FloatTensor(extracted_node).cuda()
            extracted_valid_ind = torch.IntTensor(extracted_valid_ind).cuda()

        else:
            extracted_adj = torch.FloatTensor(extracted_adj)
            extracted_node = torch.FloatTensor(extracted_node)
            extracted_valid_ind = torch.IntTensor(extracted_valid_ind)
        
        return extracted_node, extracted_adj, extracted_valid_ind


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--No', type=int, default=128)
    parser.add_argument('--edge', type=int, default=1)
    parser.add_argument('--sparse', action='store_true', default=False,
                    help='whether use sparse matrix.')
    parser.add_argument('--hetero', action='store_true', default=False,
                    help='whether using multiple edge types.')
    parser.add_argument('--used_edge', type=int, default=2,
                    help='0: using all egde. 1, 2, 3...: use only corresponding edge type.')
    args = parser.parse_args()

    adj, feature, label = load_data(args, path='./data/dblp/', dataset="dblp")
    
    dataset = RawGraph(args, adj, label, feature)

    #ipdb.set_trace()
    for i in range(adj.shape[0]//args.batch):
        #extracted_node, extracted_adj, extracted_valid_ind, extracted_label = dataset.get_tosave_batch(num=args.batch, preset_size=args.No)
        
        extracted_node, extracted_adj, extracted_valid_ind, extracted_label = dataset.get_tosave_batch(num=args.batch, preset_size=args.No)

        dataset.save_batched_pickle(i, extracted_node, extracted_adj, extracted_valid_ind, extracted_label)
    print("process transduct to induct dataset finished!!!")