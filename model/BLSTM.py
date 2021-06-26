import torch
import torch.nn as nn

class BLSTM(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.blstm = nn.LSTM(inputdim,1024,num_layers=4,bias=True,batch_first=True,dropout=0.2,bidirectional=True,proj_size=0)
        self.l1 = nn.Linear(2048, 1024)
        self.outputlayer = nn.Linear(1024, outputdim)
        #self.conv = nn.Conv2d(1024,outputdim,kernel_size=1,padding=1,bias=False)
        #self.l1 = nn.ReLU()
        #self.soft_max = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.blstm(x)
        x= x[0]
        #x = x[:, :, 0:1024] + x[:, :, 1024:2048]
        x = self.l1(x)
        out = torch.sigmoid(self.outputlayer(x))
        #x = self.l1(x)
        #out = self.soft_max(x)
        return out

if __name__ == "__main__":
    model = BLSTM(64, 4)
    #x = torch.rand(4, 100, 64)
    x = torch.rand(20,10,64)
    o = model(x)
    # (bs, time, output_dim)
    print(o.shape)