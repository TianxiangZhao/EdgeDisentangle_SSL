import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init
import random
import utils

import ipdb


#GCN layer
class GraphConvolution(Module):
    """
    Simple GCN layer, obtained from https://github.com/tkipf/pygcn/blob/master/pygcn
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)

        if not adj.is_sparse:
            output = torch.spmm(adj, support)
        else: #to support autograd
            indices = adj.coalesce().indices()
            values = adj.coalesce().values()
            output = utils.sp_matmul(indices, values.unsqueeze(-1), support)
        
        #for 3_D batch, need a loop!!!


        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


#GraphSage Layer
class SageConv(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()

        self.proj = nn.Linear(in_features*2, out_features, bias=bias)

        self.reset_parameters()

        #print("note: for dense graph in graphsage, require it normalized.")

    def reset_parameters(self):

        nn.init.normal_(self.proj.weight)

        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, features, adj):
        """
        Args:
            adj: can be sparse or dense matrix.
        """

        #fuse info from neighbors. to be added:
        if not adj.is_sparse:
            if len(adj.shape) == 3:
                neigh_feature = torch.bmm(adj, features) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1],-1))+1)
            else:
                neigh_feature = torch.matmul(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1)+1)
        else:
            #print("spmm not implemented for batch training. Note!")
            #

            #implementation to support gradient computation
            indices = adj.coalesce().indices()
            values = adj.coalesce().values()
            neigh_feature = utils.sp_matmul(indices, values.unsqueeze(-1), features) / (adj.to_dense().detach().sum(dim=1).reshape(adj.shape[0], -1)+1)

            #classical implementation
            #neigh_feature = torch.spmm(adj, features) / (adj.to_dense().detach().sum(dim=1).reshape(adj.shape[0], -1)+1)

        #perform conv
        data = torch.cat([features,neigh_feature], dim=-1)
        combined = self.proj(data)

        return combined


#Multihead self-attention layer
class MultiHeadSAN(Module):#currently, allowed for only one sample each time. As no padding mask is required.
    def __init__(
        self,
        input_dim,
        num_heads,
        kdim=None,
        vdim=None,
        embed_dim = 128,#should equal num_heads*head dim
        v_embed_dim = None,
        dropout=0.1,
        bias=True,
    ):
        super(MultiHeadSAN, self).__init__()
        self.input_dim = input_dim
        self.kdim = kdim if kdim is not None else input_dim
        self.vdim = vdim if vdim is not None else input_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.v_embed_dim = v_embed_dim if v_embed_dim is not None else embed_dim

        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        assert self.v_embed_dim % num_heads ==0, "v_embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5


        self.q_proj = nn.Linear(self.input_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.v_embed_dim, bias=bias)

        self.out_proj = nn.Linear(self.v_embed_dim, self.v_embed_dim//self.num_heads, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if True:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.normal_(self.k_proj.weight)
            nn.init.normal_(self.v_proj.weight)
            nn.init.normal_(self.q_proj.weight)
        else:
            nn.init.normal_(self.k_proj.weight)
            nn.init.normal_(self.v_proj.weight)
            nn.init.normal_(self.q_proj.weight)

        nn.init.normal_(self.out_proj.weight)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

        if self.bias:
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.q_proj.bias, 0.)

    def forward(
        self,
        query,
        key,
        value,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ):
        """Input shape: Time x Batch x Channel
        Args:
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        batch_num, node_num, input_dim = query.size()

        assert key is not None and value is not None

        #project input
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q * self.scaling

        #compute attention
        q = q.view(batch_num, node_num, self.num_heads, self.head_dim).transpose(-2,-3).contiguous().view(batch_num*self.num_heads, node_num, self.head_dim)
        k = k.view(batch_num, node_num, self.num_heads, self.head_dim).transpose(-2,-3).contiguous().view(batch_num*self.num_heads, node_num, self.head_dim)
        v = v.view(batch_num, node_num, self.num_heads, self.vdim).transpose(-2,-3).contiguous().view(batch_num*self.num_heads, node_num, self.vdim)
        attn_output_weights = torch.bmm(q, k.transpose(-1,-2))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        #drop out
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        #collect output
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.view(batch_num, self.num_heads, node_num, self.vdim).transpose(-2,-3).contiguous().view(batch_num, node_num, self.v_embed_dim)
        attn_output = self.out_proj(attn_output)


        if need_weights:
            attn_output_weights = attn_output_weights #view: (batch_num, num_heads, node_num, node_num)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output


#GraphAT layers
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward_sparse(self,input,adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        indices = adj.coalesce().indices()

        edge_h = torch.cat((h[indices[0, :], :], h[indices[1, :], :]), dim=1)
        edge_e = self.leakyrelu(torch.matmul(edge_h, self.a))

        
        attention = utils.sp_softmax(indices, edge_e, N)
        attention = F.dropout(attention, self.dropout, training=self.training)
        edge_h = F.dropout(edge_h, self.dropout, training=self.training)
        h_prime = utils.sp_matmul(indices, attention, h)

        return h_prime


    def forward_dense(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return h_prime


    def forward(self, input, adj):

        if adj.is_sparse:
            h_prime = self.forward_sparse(input, adj)
        else:
            h_prime = self.forward_dense(input, adj)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


#Disentangle GraphAT layers
class DisGALayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, att_type=1, gnn_type='AT'):
        super(DisGALayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.att_type = att_type
        self.gnn_type = gnn_type


        if self.att_type == 3:
            self.W = nn.Parameter(torch.zeros(size=(in_features*2, out_features)))#for estimating edges
            nn.init.xavier_uniform_(self.W.data, gain=1.414)
            self.a = nn.Parameter(torch.zeros(size=(out_features, 1)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)
        else:    
            self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))#for estimating edges
            nn.init.xavier_uniform_(self.W.data, gain=1.414)
            self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)


        if self.gnn_type == 'AT':
            self.W_em = nn.Parameter(torch.zeros(size=(in_features, out_features)))#for extracting output feature
            nn.init.xavier_uniform_(self.W_em.data, gain=1.414)
        elif self.gnn_type == 'SAGE':
            self.ag_layer = SageConv(in_features,out_features)
        elif self.gnn_type == 'GCN':
            self.ag_layer = GraphConvolution(in_features, out_features)


    def forward_sparse(self,input,adj, aux_indices=None):
        #aux_indices: [edge_index, edge_index, ...] list of indices for calculating attention score
        N = input.size()[0]

        indices = adj.coalesce().indices()
        if aux_indices is not None: 
            if not isinstance(aux_indices, list):
                aux_indices = [aux_indices]

        if self.att_type == 1:
            h = torch.mm(input, self.W)
            #attention via product with original
            edge_h = torch.cat((h[indices[0, :], :], h[indices[1, :], :]), dim=1)
            edge_e = torch.matmul(edge_h, self.a)

            if aux_indices is not None:
                edge_auxs = []
                for aux_indice in aux_indices:
                    edge_aux = torch.cat((h[aux_indice[0, :], :], h[aux_indice[1, :], :]), dim=1)
                    edge_aux = torch.matmul(edge_aux, self.a)
                    edge_auxs.append(edge_aux)

        elif self.att_type == 2:        
            h = torch.mm(input, self.W)
            #attention via dot-product
            edge_e = torch.mul(h[indices[0, :], :], h[indices[1, :], :]).sum(-1, keepdim=True)
            #edge_e = edge_e/np.sqrt(self.out_features)

            if aux_indices is not None:
                edge_auxs = []
                for aux_indice in aux_indices:
                    edge_aux = torch.mul(h[aux_indice[0, :], :], h[aux_indice[1, :], :]).sum(-1, keepdim=True)
                    edge_auxs.append(edge_aux)

        elif self.att_type == 3:
            edge_h = torch.cat((input[indices[0, :], :], input[indices[1, :], :]), dim=1)
            edge_h = torch.mm(edge_h, self.W)
            edge_h = F.leaky_relu(edge_h)
            #attention via product with original
            edge_e = torch.matmul(edge_h, self.a)
            
            if aux_indices is not None:
                edge_auxs = []
                for aux_indice in aux_indices:
                    edge_aux = torch.cat((input[aux_indice[0, :], :], input[aux_indice[1, :], :]), dim=1)
                    edge_aux = torch.mm(edge_aux, self.W)
                    edge_aux = F.leaky_relu(edge_aux)
                    #attention via product with original
                    edge_aux = torch.matmul(edge_aux, self.a)
                    edge_auxs.append(edge_aux)
        

        edge_ob = torch.sigmoid(edge_e)
        attention = utils.sp_softmax(indices, edge_ob, N)
        attention = F.dropout(attention, self.dropout, training=self.training)
        #edge_h = F.dropout(edge_h, self.dropout, training=self.training)
        
        if self.gnn_type == 'AT':
            h_em = torch.mm(input, self.W_em)
            h_prime = utils.sp_matmul(indices, attention, h_em)
        elif self.gnn_type == 'SAGE':
            #attention to sparse tensor
            new_adj = adj.new(indices,attention.squeeze(),size=adj.size())
            h_prime = self.ag_layer(input, new_adj)
        elif self.gnn_type == 'GCN':
            
            new_adj = adj.new(indices,attention.squeeze(),size=adj.size())
            h_prime = self.ag_layer(input, new_adj)
        else:
            print('not implemented for gnn_type {} in DISGAT'.format(self.gnn_type))
            ipdb.set_trace()


        if aux_indices is not None:
            return h_prime, edge_e, edge_auxs
        else:
            return h_prime, edge_e



    def forward_dense(self, input, adj):

        if self.att_type == 1:#attention via conv layer
            if len(input.shape)==2:
                h = torch.mm(input, self.W)
                N = h.size()[0]
                a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
                e = torch.matmul(a_input, self.a).squeeze(2)

            else:
                h = torch.matmul(input, self.W)
                N = h.size()[-2]
                a_input = torch.cat([h.repeat(1, 1, N).view(h.shape[0], N * N, -1), h.repeat(1, N, 1)], dim=-1).view(h.shape[0], N, -1, 2 * self.out_features)
                e = torch.matmul(a_input, self.a).squeeze(-1)
                

        elif self.att_type ==2:
            #attention via inner product
            if len(input.shape)==2:
                h = torch.mm(input, self.W)
                N = h.size()[0]

                e = torch.mm(h, h.t())

            else:
                h = torch.matmul(input, self.W)
                B,N,_ = h.shape
                e = torch.matmul(h, torch.transpose(h,-1,-2))

            
        elif self.att_type == 3:#attention via conv layer
            if len(input.shape)==2:
                N = input.size()[0]

                e_input = torch.cat([input.repeat(1, N).view(N * N, -1), input.repeat(N, 1)], dim=1).view(-1, 2 * self.out_features)
                h = torch.mm(e_input, self.W)
                h = F.leaky_relu(h).view(N, -1, self.out_features)
                e = torch.matmul(h, self.a).squeeze(2)


            else:
                N = input.size()[-2]

                e_input = torch.cat([input.repeat(1,1, N).view(h.shape[0],N * N, -1), input.repeat(1,N, 1)], dim=1).view(h.shape[0], -1, 2 * self.out_features)
                h = torch.matmul(e_input, self.W)
                h = F.leaky_relu(h).view(h.shape[0],N, -1, self.out_features)
                e = torch.matmul(h, self.a).squeeze(-1)
                
        e_ob = torch.sigmoid(e)
        zero_vec = -9e15*torch.ones_like(e_ob)

        attention = torch.where(adj > 0, e_ob, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        assert len(input.shape)==2, 'not checked in batch-wise training yet.'

        if self.gnn_type == 'AT':
            h_em = torch.matmul(input, self.W_em)
            h_prime = torch.matmul(attention, h_em)
        elif self.gnn_type == 'SAGE':
            h_prime = self.ag_layer(input, attention)
        elif self.gnn_type == 'GCN':
            h_prime = self.ag_layer(input, attention)
        else:
            print('not implemented for gnn_type {} in DISGAT'.format(self.gnn_type))
            ipdb.set_trace()

        return h_prime, e




    def forward(self, input, adj, aux_indices=None):
        #aux_indeces: indices of edges need to calculate attention. Only needed in the sparse case

        if adj.is_sparse:
            if aux_indices is not None:
                h_prime, attention, attention_aux  = self.forward_sparse(input, adj, aux_indices)
                if self.concat:
                    return F.elu(h_prime), attention, attention_aux
                else:
                    return h_prime, attention
            else:
                h_prime, attention  = self.forward_sparse(input, adj)
        else:
            h_prime, attention = self.forward_dense(input, adj)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention


#factor GCN layer
class DisentangleLayer(nn.Module):
    """
    implemented with reference to https://github.com/ihollywhy/FactorGCN.PyTorch
    """

    def __init__(self, in_features, out_features, concat=True, n_latent=4):
        super(DisentangleLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.n_latent = n_latent

        self.linear = nn.Linear(in_features, self.out_features)
        self.att_ls = nn.ModuleList()
        for latent_i in range(self.n_latent):
            self.att_ls.append(nn.Linear(self.out_features*2, 1))
        
        self.emb = nn.Linear(in_features, int(self.out_features//n_latent))
        assert int(self.out_features//n_latent)*n_latent == out_features, "Inconsistency in FactorGNN heads structure"

    def forward_sparse(self,input,adj):
        N = input.size()[0]
        h = self.linear(input)

        indices = adj.coalesce().indices()

        feature_heads=[]
        for latent_i in range(self.n_latent):
            #attention via product with original
            edge_h = torch.cat((h[indices[0, :], :], h[indices[1, :], :]), dim=1)

            edge_e = self.att_ls[latent_i](edge_h)
        
            edge_ob = torch.sigmoid(edge_e)
            attention = utils.sp_softmax(indices, edge_ob, N)
        
            h_em = self.emb(input)

            h_prime = utils.sp_matmul(indices, attention, h_em)
            feature_heads.append(h_prime)

        h_out = torch.cat(feature_heads, dim=-1)

        return h_out


    def forward_dense(self, input, adj):
        h = self.linear(input)
        N = h.size()[0]

        feature_heads=[]
        for latent_i in range(self.n_latent):
            if len(input.shape)==2:
                a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
                e = self.att_ls[latent_i](a_input).squeeze(2)

                e_ob = torch.sigmoid(e)

                zero_vec = -9e15*torch.ones_like(e_ob)
                attention = torch.where(adj > 0, e_ob, zero_vec)
                attention = F.softmax(attention, dim=1)

                h_em = self.emb(input)
                h_prime = torch.matmul(attention, h_em)
                feature_heads.append(h_prime)

            else:
                print('not implemented yet')
                h_prime = None

                return h_prime

        
        h_out = torch.cat(feature_heads, dim=-1)
        return h_out
        

    def forward(self, input, adj):
        if adj.is_sparse:
            h_prime  = self.forward_sparse(input, adj)
        else:
            h_prime = self.forward_dense(input, adj)

        
        return h_prime


#another implementation of GraphSage, more parameters involved
class GraphSagePoolAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, nfeats, nhid):
        super(GraphSagePoolAggregator, self).__init__()
        self.agg_fc = nn.Linear(nfeats, nhid)
        self.act = nn.ReLU()
        self.nfeats = nfeats

    def forward(self, input, adj):
        support = self.act(self.agg_fc(input))
        denormalize_adj = (adj > 0).float()
        degree_mat = torch.sum(denormalize_adj, dim=0)
        denormalize_adj = denormalize_adj / degree_mat
        output = torch.spmm(denormalize_adj.t(), support)

        return output

    def functional_forward(self, input, adj, id, weights):
        support = self.act(
            torch.mm(input, weights['agg{}.agg_fc.weight'.format(id)].t()) + weights['agg{}.agg_fc.bias'.format(id)])
        denormalize_adj = (adj > 0).float()
        degree_mat = torch.sum(denormalize_adj, dim=0)
        denormalize_adj = denormalize_adj / degree_mat
        output = torch.spmm(denormalize_adj.t(), support)

        return output

class GraphSageLayer(nn.Module):

    def __init__(self, nfeats, nhid, aggregator):
        super(GraphSageLayer, self).__init__()

        self.nfeats = nfeats
        self.aggregator = aggregator
        self.nhid = nhid
        self.weight_self = nn.Parameter(torch.FloatTensor(self.nfeats, self.nhid))
        self.weight_neigh = nn.Parameter(torch.FloatTensor(self.nhid, self.nhid))
        nn.init.xavier_uniform(self.weight_self)
        nn.init.xavier_uniform(self.weight_neigh)

    def forward(self, x, adj):
        neigh_feats = self.aggregator(x, adj)
        x = torch.mm(x, self.weight_self)
        neigh_feats = torch.mm(neigh_feats, self.weight_neigh)
        combined = F.relu(torch.cat([x, neigh_feats], dim=1))
        return combined

#graph Isomorhism Network layer
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class RelationalGraphConvLayer(Module):
    def __init__(self, input_size, output_size, num_bases, num_rel, bias=False, cuda=False):
        #currently, cuda deprecated
        super(RelationalGraphConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel
        self.cuda = cuda
        
        # R-GCN weights
        if num_bases > 0:
            self.w_bases = Parameter(torch.FloatTensor(self.num_bases, self.input_size, self.output_size))
            self.w_rel = Parameter(torch.FloatTensor(self.num_rel, self.num_bases))
        else:
            self.w = Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size))
        # R-GCN bias
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.output_size))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases.data)
            nn.init.xavier_uniform_(self.w_rel.data)
        else:
            nn.init.xavier_uniform_(self.w.data)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)
        
    def forward(self, A, X):
        
        self.w = torch.einsum('rb, bio -> rio', (self.w_rel, self.w_bases)) if self.num_bases > 0 else self.w
        # Each relations * Weight
        supports = []
        for i in range(len(A)):
            if X is not None: 
                if self.cuda:
                    #supports.append(torch.mm(torch.sparse.mm(utils.csr2tensor(A[i], self.cuda), X.cuda()), self.w[i]))
                    supports.append(torch.mm(torch.spmm(A[i], X), self.w[i]))
                else:
                    #supports.append(torch.mm(torch.sparse.mm(utils.csr2tensor(A[i], self.cuda), X), self.w[i]))
                    supports.append(torch.mm(torch.spmm(A[i], X), self.w[i]))

            else:
                #supports.append(torch.mm(utils.csr2tensor(A[i], self.cuda), self.w[i]))
                print('no x')

        out = torch.stack(supports, dim=0).sum(0)
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        
        return out

#heterogeneous attention layer
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class GATConv(nn.Module):
    """
    #Implemented to remove the dependency on DGL
    #multi-head graph attention layer
    #output size: nhid*nheads

    """
    def __init__(self, nfeat, nhid, nheads, dropout, alpha, residue=False):
        super(GATConv, self).__init__()

        self.dropout = dropout
        self.alpha = alpha
        self.residue = residue

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=self.alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        if nfeat != nhid*nheads and residue:
            self.res_fc = nn.Linear(nfeat, nhid*nheads, bias=False)
        else:
            self.register_buffer('res_fc', None)

    def forward(self, x, adj):
        x_old = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        
        if self.residue:
            if self.res_fc is not None:
                x_old = self.res_fc(x_old)
            x = x+x_old


        return x

class GINConv(torch.nn.Module):
    def __init__(self, input_dim, our_dim):
        super().__init__()
        
        self.linear = torch.nn.Linear(input_dim, our_dim)

    def forward(self, A, X):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix
        
        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        X = self.linear(X + A @ X)
        X = torch.nn.functional.relu(X)
        
        return X


class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout, alpha, residue):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, alpha, residue))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](h,g).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)


class FuseLayer(nn.Module):
    def __init__(self, args, nheads, nfeat=64, residue=0):
        super(FuseLayer,self).__init__()
        self.args = args
        self.nheads = nheads
        self.nfeat=nfeat
        self.residue_dim = residue

        #self.fuse = nn.Conv1d(in_channels=nheads, out_channels=1,kernel_size=1)
        if self.args.residue_type == 0:
            self.fuse = nn.Linear(self.nfeat*nheads+self.residue_dim, self.nfeat)
        if self.args.residue_type == 1:
            self.fuse = nn.Linear(self.nfeat*nheads+self.residue_dim, self.nfeat*2)
            self.fuse2 = nn.Linear(self.nfeat*2, self.nfeat)
        if self.args.residue_type == 2:
            self.fuse = nn.Linear(self.nfeat*nheads, self.nfeat)
            if self.residue_dim != 0:
                self.fuse2 = nn.Linear(self.residue_dim, self.nfeat)


    def forward(self, feature_list, residue=None):
        #features = torch.stack(feature_list, dim=-2)
        #feature = self.fuse(features.reshape(-1, self.nheads, feature_list[0].shape[-1])).squeeze().reshape(feature_list[0].shape)

        features = torch.cat(feature_list, dim=-1)
        
        if self.args.residue_type==0:
            if self.residue_dim !=0 and residue is not None:
                features = torch.cat([features,residue], dim=-1)        
            feature = self.fuse(features)
        elif self.args.residue_type==1:
            if self.residue_dim !=0 and residue is not None:
                features = torch.cat([features,residue], dim=-1)        
            feature = F.leaky_relu(self.fuse(features))        
            feature = self.fuse2(feature)
        elif self.args.residue_type==2:
            feature = self.fuse(features)
            if self.residue_dim !=0 and residue is not None:
                feature_res = self.fuse2(residue)
                feature = feature + feature_res

        if self.args.fuse_no_relu:
            feature = feature
        else:
            feature = F.leaky_relu(feature)

        return feature