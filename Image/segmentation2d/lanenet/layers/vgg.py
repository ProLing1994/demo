import torch
import torch.nn as nn

class VggBase(nn.Module):
    def __init__(self, channel, *args):
        super(VggBase, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        return self.conv(x)

class VggWide(nn.Module):
    def __init__(self, channel, *args):
        super(VggWide, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 5, 1, 2),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':
    from sys import path
    path.append('./')
    from utils.utils import OPcounter
    
    x = torch.rand(8, 16, 32, 32)
    net1 = VggBase(16)
    net2 = VggWide(16)
    y1 = net1(x)
    y2 = net2(x)
    print(x.shape)
    print(y1.shape)
    print(y2.shape)
    print(OPcounter(net1, x.shape[1], x.shape[2:], 'cpu', False))
    print(OPcounter(net2, x.shape[1], x.shape[2:], 'cpu', False))
