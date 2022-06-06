from numpy.testing._private.utils import assert_equal
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttentionLayer, GraphSagePoolAggregator, GraphSageLayer, SageConv, RelationalGraphConvLayer,HANLayer, DisGALayer, FuseLayer, GINConv, DisentangleLayer
import torch
from utils import init_weights
import math
import ipdb
import scipy.sparse
import numpy as np

#GCN
class GCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def get_em(self, x, adj):
        
        feature_1 = F.relu(self.gc1(x, adj))
        feature_2 = F.dropout(feature_1, self.dropout, training=self.training)
        feature_2 = F.relu(self.gc2(feature_2, adj))
        feature_2 = F.dropout(feature_2, self.dropout, training=self.training)

        return [feature_1, feature_2]

#GIN, simple implementation
##adapted from https://github.com/matiasbattocchia/gin/blob/master/GIN.ipynb

class GIN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        
        self.convs = torch.nn.ModuleList()
        
        for _ in range(n_layers):
            self.convs.append(GINConv(hidden_dim, hidden_dim))
        
        # In order to perform graph classification, each hidden state
        # [batch x nodes x hidden_dim] is concatenated, resulting in
        # [batch x nodes x hiddem_dim*(1+n_layers)], then aggregated
        # along nodes dimension, without keeping that dimension:
        # [batch x hiddem_dim*(1+n_layers)].
        self.out_proj = torch.nn.Linear(hidden_dim*(1+n_layers), output_dim)

    def forward(self, X, A):
        X = self.in_proj(X)

        hidden_states = [X]
        
        for layer in self.convs:
            X = layer(A, X)
            hidden_states.append(X)

        #X = torch.cat(hidden_states, dim=2).sum(dim=1)
        X = torch.cat(hidden_states, dim=-1)
        X = self.out_proj(X)

        return F.log_softmax(X, dim=-1)


    def get_em(self, X, A):
        X = self.in_proj(X)

        hidden_states = []
        
        for layer in self.convs:
            X = layer(A, X)
            hidden_states.append(X)

        #X = torch.cat(hidden_states, dim=2).sum(dim=1)

        return hidden_states
    

#Factor GCN, implemented with reference to https://github.com/ihollywhy/FactorGCN.PyTorch/
class FactorGCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nheads=4):
        """Dense version of GAT."""
        super(FactorGCN, self).__init__()
        self.args = args

        self.FactorNN1 = DisentangleLayer(nfeat, nhid, n_latent=nheads)
        self.FactorNN2 = DisentangleLayer(nhid, nclass, n_latent=nheads)

    def forward(self, x, adj):
        feature_1 = self.FactorNN1(x, adj)
        feature_1 = F.elu(feature_1)

        feature_2 = self.FactorNN2(feature_1, adj)

        return F.log_softmax(x, dim=1)

    
    def get_em(self, x, adj):
        feature_1 = self.FactorNN1(x, adj)
        feature_1 = F.elu(feature_1)

        feature_2 = self.FactorNN2(feature_1, adj)
        feature_2 = F.elu(feature_2)


        return [feature_1, feature_2]

#GAT
class GAT(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, alpha=0.1, nheads=4):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.args = args

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

    
    def get_em(self, x, adj):
        feature_1 = F.dropout(x, self.dropout, training=self.training)
        feature_1 = torch.cat([att(feature_1, adj) for att in self.attentions], dim=1)
        feature_2 = F.dropout(feature_1, self.dropout, training=self.training)
        feature_2 = F.elu(self.out_att(feature_2, adj))


        return [feature_1, feature_2]


#New dynamic model for pretrain
class DISGAT(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, is_specific=[True, True], alpha=0.1, nheads=4):
        """Dense version of GAT."""
        #specific layers correspond to 
        super(DISGAT, self).__init__()
        self.dropout = dropout
        self.args = args
        self.nheads = nheads
        self.gnn_type = args.gnn_type

        self.is_specific = is_specific
        
        self.attentions1 = [DisGALayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, att_type=args.att, gnn_type=self.gnn_type) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)

        self.attentions2 = [DisGALayer(nhid, nclass, dropout=dropout, alpha=alpha, concat=True, att_type=args.att, gnn_type=self.gnn_type) for _ in
                           range(nheads)]
        
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)

        if args.residue:
            self.fuser1 = FuseLayer(args, nheads, nfeat=nhid, residue=nfeat)
            self.fuser2 = FuseLayer(args,nheads, nfeat=nhid, residue=nhid)
        else:
            self.fuser1 = FuseLayer(args,nheads, nfeat=nhid)
            self.fuser2 = FuseLayer(args,nheads, nfeat=nhid)

    def forward(self, x, adj, fusers):
        if not isinstance(fusers, list):
            fusers = [fusers]

        x = F.dropout(x, self.dropout, training=self.training)

        x1_out = []
        adj1_out = []
        for i in range(self.nheads):
            x_out, adj_out = self.attentions1[i](x, adj)
            x1_out.append(x_out)
            adj1_out.append(adj_out)

        if not self.is_specific[0]:
            x = self.fuser1(x1_out, x)
        else:
            x = fusers[0](x1_out, x)

        feature = F.dropout(x, self.dropout, training=self.training)

        
        x2_out = []
        adj2_out = []
        for i in range(self.nheads):
            x_out, adj_out = self.attentions2[i](feature, adj)
            x2_out.append(x_out)
            adj2_out.append(adj_out)

        if not self.is_specific[1]:
            x = self.fuser2(x2_out, feature)
        else:
            x = fusers[1](x2_out, feature)

        return F.log_softmax(x, dim=1)

    
    def get_em(self, x, adj, fusers):
        if not isinstance(fusers, list):
            fusers = [fusers]

        x = F.dropout(x, self.dropout, training=self.training)

        x1_out = []
        adj1_out = []
        for i in range(self.nheads):
            x_out, adj_out = self.attentions1[i](x, adj)
            x1_out.append(x_out)
            adj1_out.append(adj_out)

        if not self.is_specific[0]:
            x = self.fuser1(x1_out, x)
        else:
            x = fusers[0](x1_out, x)

        feature_1 = F.dropout(x, self.dropout, training=self.training)

        
        x2_out = []
        adj2_out = []
        for i in range(self.nheads):
            x_out, adj_out = self.attentions2[i](feature_1, adj)
            x2_out.append(x_out)
            adj2_out.append(adj_out)

        if not self.is_specific[1]:
            x = self.fuser2(x2_out, feature_1)
        else:
            x = fusers[1](x2_out, feature_1)

        feature_2 = F.dropout(x, self.dropout, training=self.training)

        return [feature_1, feature_2]

    def get_adjs(self, x, adj, fusers):
        #return a list(layer number) of list(nheads), each element is an unormalized predicted attention matrix
        if not isinstance(fusers, list):
            fusers = [fusers]

        x = F.dropout(x, self.dropout, training=self.training)

        x1_out = []
        adj1_out = []
        for i in range(self.nheads):
            x_out, adj_out = self.attentions1[i](x, adj)
            x1_out.append(x_out)
            adj1_out.append(adj_out)

        if not self.is_specific[0]:
            x = self.fuser1(x1_out, x)
        else:
            x = fusers[0](x1_out, x)

        feature = F.dropout(x, self.dropout, training=self.training)

        
        x2_out = []
        adj2_out = []
        for i in range(self.nheads):
            x_out, adj_out = self.attentions2[i](feature, adj)
            x2_out.append(x_out)
            adj2_out.append(adj_out)

        if not self.is_specific[1]:
            x = self.fuser2(x2_out, feature)
        else:
            x = fusers[1](x2_out, feature)

        return [adj1_out, adj2_out]#it may be sparse or not

    def predict_adjs_sparse(self, x, adj, fusers, auxiliary_edges):
        #return a list(layer number) of list(nheads), each element is an unormalized predicted attention matrix
        #

        if not isinstance(fusers, list):
            fusers = [fusers]

        x = F.dropout(x, self.dropout, training=self.training)

        x1_out = []
        adj1_out = []
        adj_aux1 = []
        for i in range(self.nheads):
            x_out, adj_out, aux_adj_out = self.attentions1[i](x, adj, auxiliary_edges)
            x1_out.append(x_out)
            adj1_out.append(adj_out)
            adj_aux1.append(aux_adj_out)

        if not self.is_specific[0]:
            x = self.fuser1(x1_out, x)
        else:
            x = fusers[0](x1_out, x)

        feature = F.dropout(x, self.dropout, training=self.training)

        
        x2_out = []
        adj2_out = []
        adj_aux2 = []
        for i in range(self.nheads):
            x_out, adj_out, aux_adj_out = self.attentions2[i](feature, adj, auxiliary_edges)
            x2_out.append(x_out)
            adj2_out.append(adj_out)
            adj_aux2.append(aux_adj_out)

        if not self.is_specific[1]:
            x = self.fuser2(x2_out, feature)
        else:
            x = fusers[1](x2_out, feature)

        return [adj_aux1, adj_aux2]#it may be sparse or not

    
    def get_edge_em(self, x, adj, fusers):
        #return a list(layer number) of list(nheads), each element is the representations for channels
        if not isinstance(fusers, list):
            fusers = [fusers]

        x = F.dropout(x, self.dropout, training=self.training)

        x1_out = []
        edge_out = []
        for i in range(self.nheads):
            x_out, adj_out = self.attentions1[i](x, adj)
            x1_out.append(x_out)

            #get edge embedding
            edge_out.append(torch.cat((x,x_out),dim=-1))



        if not self.is_specific[0]:
            x = self.fuser1(x1_out, x)
        else:
            x = fusers[0](x1_out,x)

        feature = F.dropout(x, self.dropout, training=self.training)

        
        x2_out = []
        edge2_out = []
        for i in range(self.nheads):
            x_out, adj_out = self.attentions2[i](feature, adj)
            x2_out.append(x_out)

            edge2_out.append(torch.cat((feature,x_out),dim=-1))


        if not self.is_specific[1]:
            x = self.fuser2(x2_out, feature)
        else:
            x = fusers[1](x2_out, feature)

        return [edge_out, edge2_out]#it may be sparse or not



#GraphSage
class GraphSage(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout):
        super(GraphSage, self).__init__()
        self.agg1 = GraphSagePoolAggregator(nfeat, nhid)
        self.enc1 = GraphSageLayer(nfeat, nhid, self.agg1)
        self.agg2 = GraphSagePoolAggregator(2 * nhid, nhid)
        self.enc2 = GraphSageLayer(2 * nhid, nhid, self.agg2)
        self.dropout = dropout

        self.emb_fc = nn.Linear(2*nhid, nhid)

        self.final_fc = nn.Linear(2 * nhid, nclass)

    def forward(self, x, adj):
        x = F.normalize(self.enc1(x, adj), dim=1)
        x = self.enc2(x, adj)
        x = self.final_fc(x)
        return F.log_softmax(x, dim=1)

    
    def get_em(self, x, adj):
        feature_1 = F.normalize(self.enc1(x, adj), dim=1)
        feature_2 = self.enc2(feature_1, adj)
        feature_2 = self.emb_fc(feature_2)


        return [feature_1,feature_2]
    

#another version of GraphSage
class Sage2(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout):
        super(Sage2, self).__init__()

        self.args = args
        self.sage1 = SageConv(nfeat, nhid)
        self.sage2 = SageConv(nhid, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.mlp(x)

        return F.log_softmax(x, dim=1)

    
    def get_em(self, x, adj):
        feature_1 = F.relu(self.sage1(x, adj))
        feature_1 = F.dropout(feature_1, self.dropout, training=self.training)
        feature_2 = F.relu(self.sage2(feature_1, adj))
        feature_2 = F.dropout(feature_2, self.dropout, training=self.training)
        

        return [feature_1,feature_2]

#RGCN
class RelationalGraphConvModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_bases=0, num_rel=3, num_layer=2, dropout=0.1, featureless=False):
        super(RelationalGraphConvModel, self).__init__()
        assert num_layer >=2

        self.num_layer = num_layer
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        self.act = nn.ReLU()
        self.layers.append(RelationalGraphConvLayer(input_size, hidden_size, num_bases, num_rel, bias=False))
        for i in range(self.num_layer-1):
            if i != self.num_layer-2:
                self.layers.append(RelationalGraphConvLayer(hidden_size, hidden_size, num_bases, num_rel, bias=False))
            else:
                self.layers.append(RelationalGraphConvLayer(hidden_size, output_size, num_bases, num_rel, bias=False))
        
    def forward(self, X, A):
        x = X
        #x = None # featureless
        for i, layer in enumerate(self.layers):
            x = layer(A, x)
            if i != self.num_layer-1:
                x = F.dropout(self.act(x), self.dropout, training=self.training)
            else:
                x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)

#HAN
class HAN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_heads, num_meta_paths=3, dropout=0.1, alpha=0.1, residue=False):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout, alpha, residue))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout, alpha, residue))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, h, g):

        for gnn in self.layers:
            h = gnn(g, h)

        h = self.predict(h)

        return F.log_softmax(h, dim=1)


class MLPEncoder(nn.Module):
    def __init__(self,args, nfeat, nhid, nclass, layers=2, dropout=0.1):
        super(MLPEncoder, self).__init__()
        assert layers==2, "MLP Encoder must has exactly two layers"
        in_size = nfeat
        self.dropout = dropout

        self.layer1 = nn.Linear(in_size,nhid)
        self.layer2 = nn.Linear(nhid,nclass)

    def forward(self, x, adj):
        feature_1 = F.relu(self.layer1(x))
        feature_1 = F.dropout(feature_1, self.dropout, training=self.training)
        feature_2 = F.relu(self.layer2(feature_1))
        feature_2 = F.dropout(feature_2, self.dropout, training=self.training)

        return F.log_softmax(feature_2, dim=1)

    def get_em(self, x, adj):
        feature_1 = F.relu(self.layer1(x))
        feature_1 = F.dropout(feature_1, self.dropout, training=self.training)
        feature_2 = F.relu(self.layer2(feature_1))
        feature_2 = F.dropout(feature_2, self.dropout, training=self.training)

        return [feature_1, feature_2]
    



class MLP(nn.Module):
    def __init__(self, in_feat, hidden_size, out_size, layers=2, dropout=0.1):
        super(MLP, self).__init__()

        modules = []
        in_size = in_feat
        for layer in range(layers-1):
            modules.append(nn.Linear(in_size, hidden_size))
            in_size = hidden_size
            modules.append(nn.LeakyReLU(0.1))
        modules.append(nn.Linear(in_size, out_size))

        self.model = nn.Sequential(*modules)

    def forward(self, features, cls=False):
        output = self.model(features)

        if cls:
            return F.log_softmax(output, dim=1)
        else:
            return output


class EdgePredictor(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.1, layers=1):
        super(EdgePredictor, self).__init__()
        self.dropout = dropout
        
        self.layers = layers

        modules = []
        for layer in range(layers-1):
            modules.append(nn.Linear(nfeat, nhid))
            nfeat = nhid
            modules.append(nn.LeakyReLU(0.1))
        modules.append(nn.Linear(nfeat, nhid))
        
        self.model = nn.Sequential(*modules)

        self.de_weight = nn.Parameter(torch.FloatTensor(nhid, nhid))
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

        self.model.apply(init_weights)


    def forward(self, x_embed):
        if self.layers >= 1:
            x_out = self.model(x_embed)
        else:
            x_out = x_embed

        combine = F.linear(x_out, self.de_weight)
        if len(combine.shape)==2:
            adj_out = torch.sigmoid(torch.mm(combine, x_out.transpose(-1,-2)))
        elif len(combine.shape)==3:
            adj_out = torch.sigmoid(torch.bmm(combine, x_out.transpose(-1,-2)))
        else:
            print('Error, wrong input graph embedding dimension for edge prediction!')
            ipdb.set_trace()

        return adj_out

#H2GCN

class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class H2GCN(nn.Module):
    """ our implementation, from https://github.com/CUAI/Non-Homophily-Benchmarks """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                    num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                    use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        self.feature_embed = MLP(in_channels, hidden_channels,
                hidden_channels, layers=num_mlp_layers, dropout=dropout)


        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')

        self.projects=[]
        for l in range(num_layers):
            last_dim = hidden_channels*(2**(l+1))
            self.projects.append(nn.Linear(last_dim, out_channels))
        
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.last_preject = nn.Linear(last_dim, out_channels)


        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        adj_t = edge_index
        adj_t = adj_t.to_dense()
        ind = np.diag_indices(adj_t.shape[0])
        adj_t[ind[0], ind[1]] = torch.zeros(adj_t.shape[0])

        adj_t2 = torch.matmul(adj_t, adj_t)
        adj_t2[ind[0], ind[1]] = torch.zeros(adj_t.shape[0])

        adj_t = scipy.sparse.csr_matrix(adj_t.cpu().numpy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.cpu().numpy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)



    def forward(self, x, adj):
        n = x.shape[0]

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        x = self.feature_embed(x)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.last_project(x)

        return x

    def get_em(self, x, adj):
        n = x.shape[0]

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        x = self.feature_embed(x)
        x = self.activation(x)
        xs = []
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(self.projects[i](x))

        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        xs.append(self.projects[-1](x))

        return xs

#Mixhop
class MixHopLayer(nn.Module):
    """ Our MixHop layer """
    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops+1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x) ]
        for j in range(1,self.hops+1):
            # less runtime efficient but usually more memory efficient to mult weight matrix first
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = torch.matmul(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)

class MixHop(nn.Module):
    """ our implementation of MixHop, from https://github.com/CUAI/Non-Homophily-Benchmarks
    some assumptions: the powers of the adjacency are [0, 1, ..., hops],
        with every power in between
    each concatenated layer has the same dimension --- hidden_channels
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, hops=2):
        super(MixHop, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))
        for _ in range(num_layers - 2):
            self.convs.append(
                MixHopLayer(hidden_channels*(hops+1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))

        self.convs.append(
            MixHopLayer(hidden_channels*(hops+1), out_channels, hops=hops))

        # note: uses linear projection instead of paper's attention output
        self.final_project = nn.Linear(out_channels*(hops+1), out_channels)

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, x, adj):
        n = x.shape[0]
        adj_t = adj
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        x = self.final_project(x)
        return x

        
    def get_em(self, x, adj):
        n = x.shape[0]
        adj_t = adj
        
        feat = []

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            feat.append(x)

        x = self.convs[-1](x, adj_t)

        feat.append(x)

        assert len(feat)==2, 'configure encoder layer to 2'
        return feat

