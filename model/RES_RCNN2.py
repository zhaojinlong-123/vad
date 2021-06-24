import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

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

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class CRNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.dp = nn.Dropout(0.3)
        self.lp = nn.LPPool2d(4, (1, 16))
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.hs1 = hswish()
        self.resbk1 = Block(3,32,32, 64,nn.ReLU(inplace=True), None,2)
        self.resbk2 = Block(3, 64, 64, 128, nn.ReLU(inplace=True), None, 2)
        self.resbk3 = Block(3, 128, 128, 128, nn.ReLU(inplace=True), None, 1)
        self.gru = nn.GRU(128,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.outputlayer = nn.Linear(256, outputdim)
        self.outputlayer.apply(init_weights)
        self.conv1.apply(init_weights)
        self.resbk1.apply(init_weights)
        self.resbk2.apply(init_weights)
        self.resbk3.apply(init_weights)

    def forward(self, x):
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x= self.hs1(self.bn1(self.conv1(x)))
        x = self.resbk1(x)
        x = self.resbk2(x)
        x = self.resbk3(x)
        x = self.lp(x)
        x =self.dp(x)
        #x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        decision_time = torch.nn.functional.interpolate(
            decision_time.transpose(1, 2),
            time,
            mode='linear',
            align_corners=False).transpose(1, 2)
        return decision_time


if __name__ == "__main__":
    model = CRNN(64, 4)
    #x = torch.rand(4, 100, 64)
    x = torch.rand(1,100,64)
    o = model(x)
    # (bs, time, output_dim)
    print(o.shape)