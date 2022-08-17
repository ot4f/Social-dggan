import torch.nn as nn
import torch.nn.functional as F

class LSTM_Discriminator(nn.Module):
    def __init__(self, input_size, embedding_size=64, hidden_size=64, mlp_size=1024):
        super(LSTM_Discriminator, self).__init__()

        self.spatial_embedding = nn.Linear(input_size, embedding_size)
        self.encoder = nn.LSTM(
            embedding_size, 
            hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, 1)
        )
    
    def forward(self, x):
        """
        :param x, Tensor of shape (batch, seq, node, feat)
        """
        batch, seq_len, node_num, _ = x.shape
        x = x.permute(0,2,1,3)
        x = x.reshape(batch*node_num, seq_len, -1)
        x = self.spatial_embedding(x)
        _, (x, _) = self.encoder(x)
        output = self.fc(x)
        output = output.reshape(batch, node_num)
        return output
