from torch import nn
import torch
from einops import rearrange
class DNN(nn.Module):
    def __init__(self,inputdim, outputdim,first_hidden_features=512,second_hidden_features=512,dropout=0.2):
        super(DNN, self).__init__()
        self.d1 = nn.Dropout(p=dropout)
        self.l1 = nn.Linear(inputdim, first_hidden_features)
        self.b1 = nn.BatchNorm1d(first_hidden_features)
        self.r1 = nn.ReLU()
        self.d2 = nn.Dropout(p=dropout)
        self.l2 = nn.Linear(first_hidden_features, second_hidden_features)
        self.b2 = nn.BatchNorm1d(second_hidden_features)
        self.r2 = nn.ReLU()
        self.d3 = nn.Dropout(p=dropout)
        self.l3 = nn.Linear(second_hidden_features, outputdim)

        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, features):
        # features: (batch_size, window_size, feature_size)
        x = self.d1(features)
        x = self.l1(x)
        x = rearrange(x, "bs time dim-> bs dim time")
        x = self.b1(x)
        x = rearrange(x, "bs time dim-> bs dim time")
        x = self.r1(x)
        x = self.d2(x)
        x = self.l2(x)
        x = rearrange(x, "bs time dim-> bs dim time")
        x = self.b2(x)
        x = rearrange(x, "bs time dim-> bs dim time")
        x = self.r2(x)
        x = self.d3(x)
        x = self.l3(x)

        x = self.log_softmax(x)

        return x

if __name__ == "__main__":
    model = DNN(64,4)
    #x = torch.rand(4, 100, 64)
    x = torch.rand(20,10,64)
    o = model(x)
    # (bs, time, output_dim)
    print(o.shape)