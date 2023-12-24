import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a_SS = nn.Parameter(torch.empty(size=(2 * out_features, 1))) # same speaker
        nn.init.xavier_uniform_(self.a_SS.data, gain=1.414)
        self.a_DS = nn.Parameter(torch.empty(size=(2 * out_features, 1))) # different speaker
        nn.init.xavier_uniform_(self.a_DS.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.csk_linear = nn.Linear(out_features, out_features)
        

    def forward(self, h, h_xReact, h_oReact, utt_xReact_mask, utt_oReact_mask):
        bz, N, _ = h.size()
        Wh = torch.matmul(h, self.W)  # linearly transform the node's feature vector. h.shape: (N, in_features)
        h_xReact = self.csk_linear(h_xReact)
        h_oReact = self.csk_linear(h_oReact)
        
        a_input_SS = torch.cat([Wh.repeat(1,1,N).view(-1, N*N, self.out_features), (Wh + h_xReact).repeat(1, N, 1)], dim=-1).view(-1, N, N, 2*self.out_features)
        a_input_DS = torch.cat([Wh.repeat(1,1,N).view(-1, N*N, self.out_features), (Wh + h_oReact).repeat(1, N, 1)], dim=-1).view(-1, N, N, 2*self.out_features)
        
        e_SS = self.leakyrelu(torch.matmul(a_input_SS, self.a_SS).squeeze(-1)) # edge weight
        utt_xReact_mask, utt_oReact_mask = utt_xReact_mask[:, :N, :N], utt_oReact_mask[:, :N, :N]
        e_SS = e_SS * utt_xReact_mask
        e_DS = self.leakyrelu(torch.matmul(a_input_DS, self.a_DS).squeeze(-1))
        e_DS = e_DS * utt_oReact_mask
        e = e_SS + e_DS
        adj = utt_xReact_mask + utt_xReact_mask
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1) # normalize the edge weights
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # (N, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0) # Repeat each feature vector N times: e1, e1, ..., e1, e2, e2, ..., e2, ..., eN, eN, ..., eN
        Wh_repeated_alternating = Wh.repeat(N, 1) # Repeat the feature matrix N times：e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # shape == (N * N, out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1) # Combine each node with all other nodes: (N * N, 2 * out_features)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention) 

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False) 


    def forward(self, h, h_xReact, h_oReact, utt_xReact_mask, utt_oReact_mask):
        x = F.dropout(h, self.dropout, training=self.training)
        h_xReact = F.dropout(h_xReact, self.dropout, training=self.training)
        h_oReact = F.dropout(h_oReact, self.dropout, training=self.training)
        x = torch.cat([att(x, h_xReact, h_oReact, utt_xReact_mask, utt_oReact_mask) for att in self.attentions], dim=-1) # Concatenate multiple different features obtained from multiple attention mechanisms on the same node to form a long feature.
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, h_xReact, h_oReact, utt_xReact_mask, utt_oReact_mask))
        return x
