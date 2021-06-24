import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Function


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super(DNN, self).__init__()
        hidden_size = 32
        self.fc1 = nn.Linear(inputdim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1_drop = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2_drop = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc3_drop = nn.Dropout(p=0.2)
        
        self.last = nn.Linear(hidden_size, outputdim)

        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc3.apply(init_weights)
        self.last.apply(init_weights)
    
    def forward(self, x):
        out = F.relu(self.bn1((self.fc1(x))))
        out = F.relu(self.bn2((self.fc2(out))))
        out = F.relu(self.bn3((self.fc3(out))))
        
        out = self.last(out)
        
        return out

if __name__ == "__main__":
    model = DNN(64, 4)
    #x = torch.rand(4, 100, 64)
    x = torch.rand(10,64)
    o = model(x)
    # (bs, time, output_dim)
    print(o.shape)