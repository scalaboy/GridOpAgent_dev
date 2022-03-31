import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops  #, softmax
from torch_scatter import scatter
import math
#from torch_geometric.data import data as D


class GRUGate(nn.Module):
    def __init__(self, d_model):
        super(GRUGate,self).__init__()

        self.linear_w_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_z = nn.Linear(d_model, d_model)
        self.linear_u_z = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_g = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_g = nn.Linear(d_model, d_model, bias=False)

        self.init_bias()

    def init_bias(self):
        with torch.no_grad():
            self.linear_w_z.bias.fill_(-2)

    def forward(self, x, y):
        z = torch.sigmoid(self.linear_w_z(y) + self.linear_u_z(x))
        r = torch.sigmoid(self.linear_w_r(y) + self.linear_u_r(x))
        h_hat = torch.tanh(self.linear_w_g(y) + self.linear_u_g(r*x))
        return (1.-z)*x + z*h_hat


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, dropout=0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.ln= nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_k, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.gate = GRUGate(d_model)
            
    def forward(self, x, adj):
        
        residual = x
        x = self.ln(x)
        #print('in MultiHeadAttention forward ',residual.shape,x.shape,residual,x)
        q = x
        k = x
        v = x

        d_k, n_head = self.d_k, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_k) # (n*b) x lv x dv

        adj = adj.unsqueeze(1).repeat(1, n_head, 1, 1).reshape(-1, len_q, len_q)
        output = self.attention(q, k, v, adj)
        output = output.view(n_head, sz_b, len_q, d_k)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
    
        output = F.relu(self.dropout(self.fc(output)))
        output = self.gate(residual,output)
        #print('in MultiHeadAttention forward ',residual.shape,x.shape,output.shape)
        return output  


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, dhid, dropout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, dhid)
        self.w_2 = nn.Linear(dhid, d_in)
        self.ln = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.gate = GRUGate(d_in)
            
    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = F.relu(self.w_2(F.relu((self.w_1(x)))))
        re = self.gate(residual, x)
        print('in PositionwiseFeedForward ',residual.shape,re.shape)
        return self.gate(residual, x)
        
        
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, adj):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        print('in scaled',attn.shape,adj.shape)
        attn = attn.masked_fill(adj == 0, -np.inf)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output


def get_edge_from_distance(distance_mx,DEVICE):
#def get_edge_from_distance(distance_mx):
    node_n ,node_m = distance_mx.shape
    edge_index=[[n,m] for n in range(node_n) for m in range(node_m) if(distance_mx[n][m]>0) ]
    edge_attr=[[distance_mx[n][m]] for n in range(node_n) for m in range(node_m) if(distance_mx[n][m]>0) ]
    edge_index =torch.tensor(np.array(edge_index),dtype=torch.long)
    shape_a ,shape_b = edge_index.shape
    edge_index = edge_index.view((shape_b,shape_a)).to(DEVICE)
    edge_attr =torch.tensor(edge_attr,dtype=torch.float32).to(DEVICE)
    #edge_attr =torch.tensor(np.array(edge_attr),dtype=torch.float32).to(DEVICE)
    return edge_index,edge_attr


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
def softmax_sx(src, index, num_nodes):
    """
    Given a value tensor: `src`, this function first groups the values along the first dimension
    based on the indices specified in: `index`, and then proceeds to compute the softmax individually for each group.
    """
    '''
    print('src', src)
    print('index', index)
    print('num_nodes', num_nodes)
    '''
    N = int(index.max()) + 1 if num_nodes is None else num_nodes
    out = src - scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
    #print('out', out)
    out = out.exp()
    #print('out', out)
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]

    return out / (out_sum + 1e-16)

class EdgeBase_GATConv(MessagePassing):
    def __init__(self,
                 edge_index, edge_attr,
                 in_channels,
                 out_channels,
                 edge_dim,  # new
                 heads=1,
                 negative_slope=0.2,
                 dropout=0.,
                 bias=True):
        super(EdgeBase_GATConv, self).__init__(node_dim=0, aggr='add')  # "Add" aggregation.
        self.edge_index=edge_index
        self.edge_attr=edge_attr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim  # new
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))    # emb(in) x [H*emb(out)]
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))   # 1 x H x [2*emb(out)+edge_dim]    # new
        self.edge_update1 = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))   # [emb(out)+edge_dim] x emb(out)  # new

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.edge_update1)  # new
        zeros(self.bias)

    #def forward(self, x, edge_index, edge_attr, size=None):
    def forward(self, x, size=None):
        # 1. Linearly transform node feature matrix (XΘ)
        #print('hi boy ,just run1 ',x.shape,self.weight.shape)
        batch_size, num_of_vertices, in_channels = x.shape
        pconv=torch.nn.Linear(in_channels, 1).to(self.edge_index.device)
        #pconv=torch.nn.Linear(num_of_timesteps, 1).to(self.edge_index.device)
        #print('devcice-------- ',self.edge_index.device,x.device)
        #x=pconv(x).to(self.edge_index.device).reshape(batch_size, num_of_vertices, num_of_timesteps)
        #x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        #x = torch.flatten(x, start_dim = 1)
        x=x[0]
        #print('hi boy ,just run 2',x.shape,self.weight.shape)
        
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)   # N x H x emb(out)
        #print('x in forward', x.shape)

        # 2. Add self-loops to the adjacency matrix (A' = A + I)
        '''
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)   # 2 x E
            print('edge_index1', edge_index.shape)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))   # 2 x (E+N)
            print('edge_index2', edge_index.shape)
        '''
        # 2.1 Add node's self information (value=0) to edge_attr
        #self_loop_edges = torch.zeros(x.size(0), self.edge_attr.size(1)).to(self.edge_index.device)   # N x edge_dim   # new
        #print('self_loop_edges', self_loop_edges)
        #edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)  # (E+N) x edge_dim  # new
        #print('edge_attr', edge_attr)

        # 3. Start propagating messages
        #print('in begin propagate is  ',self.edge_index.shape,x.shape,self.edge_attr.shape,size)
        return self.propagate(self.edge_index, x=x, edge_attr=self.edge_attr, size=size)  # new
                            # 2 x (E+N), N x H x emb(out), (E+N) x edge_dim, None

    def message(self, x_i, x_j, size_i, edge_index_i, edge_attr):  # Compute normalization (concatenate + softmax)
        # x_i, x_j: after linear x and expand edge (N+E) x H x emb(out)
        # = N x emb(in) @ emb(in) x [H*emb(out)] (+) E x H x emb(out)
        # edge_index_i: the col part of index  [E+N]
        # size_i: number of nodes
        # edge_attr: edge values of 1->0, 2->0, 3->0.   (E+N) x edge_dim

        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)  # (E+N) x H x edge_dim  # new
        #print('edge_attr', edge_attr.shape,x_j.shape,x_i.shape)
        x_j = torch.cat([x_j, edge_attr], dim=-1)  # (E+N) x H x (emb(out)+edge_dim)   # new
        #print('x_j', x_j.shape)

        x_i = x_i.view(-1, self.heads, self.out_channels)  # (E+N) x H x emb(out)
        #print('x_i', x_i.shape,self.att.shape)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # (E+N) x H
        #print('alpha', alpha)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        #print('hi boy ,lets go ====alpha', alpha)
        alpha = softmax_sx(alpha, edge_index_i, num_nodes=size_i)   # Computes a sparsely evaluated softmax
        #print('alpha', alpha)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        #print(f'x_j*alpha {x_j * alpha.view(-1, self.heads, 1)}')
        return x_j * alpha.view(-1, self.heads, 1)   # (E+N) x H x (emb(out)+edge_dim)

    def update(self, aggr_out):   # 4. Return node embeddings (average heads)
        # for Node 0: Based on the directed graph, Node 0 gets message from three edges and one self_loop
        # for Node 1, 2, 3: since they do not get any message from others, so only self_loop

        #print('aggr_out', aggr_out)   # N x H x (emb(out)+edge_dim)
        aggr_out = aggr_out.mean(dim=1)
        #print('aggr_out', aggr_out)   # N x (emb(out)+edge_dim)
        #print('self.edge_update', self.edge_update)   # (emb(out)+edge_dim) x emb(out)
        aggr_out = torch.mm(aggr_out, self.edge_update1)
        #print('aggr_out', aggr_out.shape)   # N x emb(out)  # new

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        #print('aggr_out',aggr_out.shape)
        return aggr_out.view(1, 177, self.out_channels)#.reshape(1,11421,4)

class Edge_GNNNet(torch.nn.Module):
    def __init__(self,edge_index, edge_attr,fea_ch):
        super(Edge_GNNNet, self).__init__()
        self.conv1 = EdgeBase_GATConv(edge_index, edge_attr,128, 128, 1)
        self.lin0 = nn.Sequential(torch.nn.Linear(256,512),nn.BatchNorm1d(512),nn.ReLU(True))
        self.lin1 = nn.Sequential(torch.nn.Linear(512,128),nn.BatchNorm1d(128),nn.ReLU(True))
        self.lin2 = nn.Sequential(torch.nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(True))
        self.lin3 = nn.Sequential(torch.nn.Linear(64,16),nn.BatchNorm1d(16),nn.ReLU(True))
        self.lin4 = nn.Sequential(torch.nn.Linear(16, 1))
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self,x):
        #x, edge_index, edge_attr = data.x,data.edge_index,data.edge_attr
        #print('input x shape is x.shape',x.shape)
        batch_size, num_of_vertices, in_channels = x.shape
        re_list=[]
        re=self.conv1(x)
        '''
        for step in range(num_of_timesteps):
            re_list.append(self.conv1(x[:,:,step,:]).view(num_of_vertices,1,in_channels))
        re = torch.cat(re_list,dim=1).view(batch_size, num_of_vertices, num_of_timesteps, in_channels)
        '''
        #print('Edge_GNNNet shape is ',re.shape)
        return re

