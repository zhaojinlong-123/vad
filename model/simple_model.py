import torch
import numpy as np
from pathlib import Path
import torch.nn as nn


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



class CRNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.outputlayer_o = nn.Linear(inputdim, 20)
        self.outputlayer = nn.Linear(20, outputdim)
        self.outputlayer.apply(init_weights)
        self.outputlayer_o.apply(init_weights)
    def forward(self, x):
        batch, time, dim = x.shape
        decision_time = torch.sigmoid(self.outputlayer(self.outputlayer_o(x))).clamp(1e-7, 1.)
        decision_time = torch.nn.functional.interpolate(
            decision_time.transpose(1, 2),
            time,
            mode='linear',
            align_corners=False).transpose(1, 2)
        return decision_time


if __name__ == "__main__":
    model = CRNN(64, 4)
    #x = torch.rand(4, 100, 64)
    x = torch.rand(1,10,64)
    o = model(x)
    # (bs, time, output_dim)
    print(o.shape)
