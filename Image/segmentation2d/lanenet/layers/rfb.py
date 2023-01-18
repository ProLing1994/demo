import torch
import torch.nn as nn

class RFBResBlock(nn.Module):
    def __init__(self, channel, *args):
        super(RFBResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel//4, 1),
            nn.BatchNorm2d(channel//4),
            nn.LeakyReLU(0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, channel//4, 3, dilation=3, padding=3),
            nn.BatchNorm2d(channel//4),
            nn.LeakyReLU(0.1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channel, channel//4, 3, dilation=5, padding=5),
            nn.BatchNorm2d(channel//4),
            nn.LeakyReLU(0.1),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(channel, channel//4, 3, dilation=7, padding=7),
            nn.BatchNorm2d(channel//4),
            nn.LeakyReLU(0.1),
        )
        self.convcat = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        xc = torch.cat([x1, x3, x5, x7], 1)
        x = self.convcat(xc) + x
        return x

class RFBResBlockPlus(nn.Module):
    def __init__(self, channel, h, w): # shape: (b, c, h, w)
        super(RFBResBlockPlus, self).__init__()
        self.pars = nn.Parameter(torch.randn((1, channel//4, h, w), requires_grad=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel//4, 1),
            nn.BatchNorm2d(channel//4),
            nn.LeakyReLU(0.1),
        )
        self.conv3 = nn.Sequential(
            # nn.ReflectionPad2d(3),
            nn.Conv2d(channel, channel//4, 3, dilation=3, padding=3),
            nn.BatchNorm2d(channel//4),
            nn.LeakyReLU(0.1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channel, channel//4, 3, dilation=5, padding=5),
            nn.BatchNorm2d(channel//4),
            nn.LeakyReLU(0.1),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(channel, channel//4, 3, dilation=7, padding=7),
            nn.BatchNorm2d(channel//4),
            nn.LeakyReLU(0.1),
        )
        
        self.convcat = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel//4*5, channel, 3),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        b, c, h, w = x.shape
        pars = self.pars.repeat(b, 1,1,1)
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        xc = torch.cat([x1, x3, x5, x7, pars], 1)
        x = self.convcat(xc) + x
        return x

class RFBResBlockFree(nn.Module):
    def __init__(self, channel, *args):
        super(RFBResBlockFree, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel//2, 1, groups=2),
            nn.BatchNorm2d(channel//2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel//2, 3, dilation=1, padding=1, groups=2),
            nn.BatchNorm2d(channel//2),
            nn.ReLU(),
            nn.Conv2d(channel//2, channel//2, 1, dilation=1, padding=0, groups=2),
            nn.BatchNorm2d(channel//2),
            nn.ReLU(),
        )
        self.convcat = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        xc = torch.cat([x1, x2], 1)
        x = self.convcat(xc) + x
        return x

if __name__ == '__main__':
    from sys import path
    path.append('./')
    from utils.utils import OPcounter
    
    x = torch.rand(8, 16, 32, 32)
    net1 = RFBResBlock(16)
    net2 = RFBResBlockPlus(16, 32, 32)
    net3 = RFBResBlockFree(16)
    
    y1 = net1(x)
    y2 = net2(x)
    y3 = net3(x)
    
    print(x.shape)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)

    print(OPcounter(net1, x.shape[1], x.shape[2:], 'cpu', False))
    print(OPcounter(net2, x.shape[1], x.shape[2:], 'cpu', False))
    print(OPcounter(net3, x.shape[1], x.shape[2:], 'cpu', False))
    
    