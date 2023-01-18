import torch
import torch.nn as nn
    
class ResBlockBase(nn.Module):
    def __init__(self, channel, *args):
        # return x + f(x)
        super(ResBlockBase, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        x = self.conv(x) + x
        return x

class ResBlockBase2(nn.Module):
    def __init__(self, channel, *args):
        # return x + f2(f1(x))
        super(ResBlockBase2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        x = self.conv1(self.conv2(x)) + x
        return x
    
class ResBlock3(nn.Module):
    def __init__(self, channel, *args):
        # return x + f2(f1(x))
        super(ResBlock3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        x = self.conv1(self.conv2(x)) + x
        return x
    
    

class ResBlockDense(nn.Module):
    def __init__(self, channel, *args):
        # 
        super(ResBlockDense, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        x0 = x.clone()
        x1 = self.conv1(x0)
        x2 = x0 + x1
        x3 = self.conv2(x2)
        x = x0 + x1 + x2 + x3
        return x

class ResBlockEfficient(nn.Module):
    def __init__(self, channel, *args):
        # 1x1 channel//2
        super(ResBlockEfficient, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel//2, 3),
            nn.BatchNorm2d(channel//2),
            nn.LeakyReLU(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel//2, channel//2, 1, bias=False),
            nn.BatchNorm2d(channel//2),
            nn.LeakyReLU(0.1),
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel//2, channel, 3),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        x0 = x.clone()
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = x0 + x3
        return x



if __name__ == '__main__':
    from sys import path
    path.append('./')
    from utils.utils import OPcounter
    
    x = torch.rand(8, 16, 32, 32)
    net1 = ResBlockBase(16)
    net2 = ResBlockBase2(16)
    net3 = ResBlockDense(16)
    net4 = ResBlockEfficient(16)
    
    y1 = net1(x)
    y2 = net2(x)
    y3 = net3(x)
    y4 = net4(x)
    
    print(x.shape)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)
    print(y4.shape)

    print(OPcounter(net1, x.shape[1], x.shape[2:], 'cpu', False))
    print(OPcounter(net2, x.shape[1], x.shape[2:], 'cpu', False))
    print(OPcounter(net3, x.shape[1], x.shape[2:], 'cpu', False))
    print(OPcounter(net4, x.shape[1], x.shape[2:], 'cpu', False))



