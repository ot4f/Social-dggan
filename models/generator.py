import torch
import torch.nn as nn
from .layers import *

class CNN_Generator(nn.Module):
    def __init__(self, STG, window_size, n_pred, in_features, out_features, out_size=2, embedding_dim=64, n_stgcnn=1, n_txpcnn=5, **kwargs):
        super(CNN_Generator, self).__init__()
        self.window_size = window_size
        self.pred_len = n_pred
        self.in_features = in_features
        self.out_features = out_features
        self.out_size = out_size
        self.embdding_dim = embedding_dim
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
        
        self.spatial_embedding = nn.Linear(self.in_features, self.embdding_dim)
        self.stgcns = nn.ModuleList()
        self.stgcns.append(STG(self.window_size, self.embdding_dim, out_features, 3))
        for j in range(1, self.n_stgcnn):
            self.stgcns.append(STG(window_size, out_features, out_features, 3))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(window_size, self.pred_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(self.pred_len, self.pred_len, 3, padding=1))

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())
        self.spatial_decoder = nn.Conv2d(self.out_features, self.out_size, 3, 1, 1)

    def forward(self, V_obs, A, att_mask=None):
        """
        :param V_obs, shape [batch_Size, n_his, node_num, in_feat]
        :param att_mask, shape [batch, node, seq]
        :return shape [batch_size, n_pred, node_num, inf_feat]
        """
        batch_size, window_size, node_num, in_feature = V_obs.shape
        noise = (torch.rand(batch_size, window_size, node_num, node_num, device=A.device))  # 构造随机的邻接矩阵
        noise = noise * (noise > 0.5)
        adjacency = A * noise

        v = self.spatial_embedding(V_obs.view(-1, in_feature)).view(batch_size, window_size, node_num, -1)
        for k in range(self.n_stgcnn):
            v = self.stgcns[k](v, adjacency, att_mask).contiguous()
        stgcn_output = v.permute(0, 1, 3, 2).contiguous()  # [batch, n_his, in_feature, node_num]
        v = self.prelus[0](self.tpcnns[0](stgcn_output))
        for k in range(1, self.n_txpcnn):
            v = self.prelus[k](self.tpcnns[k](v)) + v
        output = self.spatial_decoder(v.permute(0, 2, 1, 3))  # [batch, in_feature, n_pred, node_num]
        output = output.permute(0, 2, 3, 1)
        return output

class CNN_GAT_Generator(CNN_Generator):
    def __init__(self, **kwargs):
        super(CNN_GAT_Generator, self).__init__(STGAT, **kwargs)