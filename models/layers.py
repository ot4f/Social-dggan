import torch
import torch.nn as nn
from config import device


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, window_size, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout)
        self.elu = nn.ELU()
        self.attention_list = []

    def forward(self, adj, inp, att_mask):
        """
        inp: input_fea [batch_size, window_size, N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[window_size, N, N]
        att_mask: [batch, node, seq]
        return: shape (batch_size, window_size, N, out_features]
        """
        batch_size, window_size, N, in_features = inp.shape
        inp = inp.view(-1, in_features)
        h = torch.mm(inp, self.W)
        h = h.view(-1, N, self.out_features)

        # [batch_size, N, N, 2*out_features]
        a_input = torch.cat([h.repeat(1, 1, N).view(h.shape[0], N * N, -1), h.repeat(1, N, 1)], dim=2)\
            .view(batch_size, window_size, N, -1, 2 * self.out_features)
        # [batch_size, N, N, 1] => [batch_size, N, N]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        # [batch_size, window_size, N, N]
        e = e.view(-1, window_size, N, N)

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e * adj, zero_vec)  # [batch_size, window_size, N, N]
        attention = attention.view(-1, N, N) # [batch_size*window_size, N, N]
        attention = self.softmax(attention)
        if att_mask is not None:
            att_mask = att_mask.permute(0,2,1)
            attention = attention * torch.matmul(att_mask.unsqueeze(3), att_mask.unsqueeze(2)).reshape(-1,N,N)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)  # [batch_size, N, N].[batch_size, N, out_features] => [batch_size, N, out_features]
        if self.concat:
            output = self.elu(h_prime)
        else:
            output = h_prime
        output = output.view(-1, window_size, N, self.out_features)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolutionLayer(nn.Module):

    def __init__(self, window_size, in_features, out_features, **kwargs):
        super(GraphConvolutionLayer, self).__init__()
        self.weights = nn.Parameter(
            torch.Tensor(window_size, in_features, out_features)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)

    def forward(self, adjacency, nodes, att_mask=None):
        """
        :param adjacency: FloatTensor (batch_size, window_size, node_num, node_num)
        :param nodes: FloatTensor (batch_size, window_size, node_num, in_features)
        :return output: FloatTensor (batch_size, window_size, node_num, out_features)
        """
        batch_size = adjacency.size(0)
        window_size, in_features, out_features = self.weights.size()
        weights = self.weights.unsqueeze(0).expand(batch_size, window_size, in_features, out_features)
        output = adjacency.matmul(nodes).matmul(weights)
        return output


class STGLayer(nn.Module):
    def __init__(self, GL, window_size, in_channels, out_channels, Kt, dropout=0, residual=True, **kwargs):
        super(STGLayer, self).__init__()
        padding = ((Kt - 1) // 2, 0)

        self.gl = GL(window_size, in_channels, out_channels, dropout=0.5, alpha=0.2)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (Kt, 1),
                (1, 1),
                padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.BatchNorm2d(out_channels)
            )
        self.prelu = nn.PReLU()

    def forward(self, x, A, att_mask=None):
        """
        :param x: tensor, shape [batch_size, seq, node_num, in_feat]
        """
        x_input = x.permute(0, 3, 1, 2)
        res = self.residual(x_input)
        x = self.gl(A, x, att_mask)
        x = x.permute(0, 3, 1, 2)
        x = self.tcn(x) + res
        x = self.prelu(x)
        x = x.permute(0, 2, 3, 1)
        return x


class STGCN(STGLayer):
    def __init__(self, *args, **kwargs):
        super(STGCN, self).__init__(GraphConvolutionLayer, *args, **kwargs)

class STGAT(STGLayer):
    def __init__(self, *args, **kwargs):
        super(STGAT, self).__init__(GraphAttentionLayer, *args, **kwargs)