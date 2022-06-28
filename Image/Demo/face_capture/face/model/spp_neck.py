import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os


class Conv(nn.Module):
    def __init__(self,c1,c2,k,s=1,p=0,d=1,g=1):
        super(Conv,self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1,c2,k,stride=s,padding=p,dilation=d,groups=g),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.convs(x)

class Bottleneck(nn.Module):
    def __init__(self,c1,c2,shortcut=True,g=1,e=0.5):
        super(Bottleneck,self).__init__()
        c_ = int(c2*e)
        self.cv1 = Conv(c1,c_,k=1)
        self.cv2 = Conv(c_,c2,k=3,p=1,g=g)
        self.add = shortcut and c1 == c2

    def forward(self,x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    def __init__(self,c1,c2,n=1,shortcut=True,g=1,e=0.5):
        super(BottleneckCSP,self).__init__()
        c_ = int(c2*e)
        self.cv1 = Conv(c1,c_,k=1)
        self.cv2 = nn.Conv2d(c1,c_,kernel_size=1,bias=False)
        self.cv3 = nn.Conv2d(c_,c_,kernel_size=1,bias=False)
        self.cv4 = Conv(2*c_,c2,k=1)
        self.bn = nn.BatchNorm2d(2*c_)
        self.act = nn.ReLU(inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_,c_,shortcut,g,e=1.0) for _ in range(n)])

    def forward(self,x):
        #print(x.shape)
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1,y2),dim=1))))


class SPP(nn.Module):
    def __init__(self):
        super(SPP,self).__init__()

    def forward(self, x):
        x_1 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = F.max_pool2d(x_2, 5, stride=1, padding=2)
        x_3 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_3 = F.max_pool2d(x_3, 5, stride=1, padding=2)
        x_3 = F.max_pool2d(x_3, 5, stride=1, padding=2)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return x

class SPP_Neck(nn.Module):
    def __init__(self,):
        super(SPP_Neck,self).__init__()
        # self.conv = Conv(128, 32, k=1)  
        # self.spp = SPP()
        self.csp = BottleneckCSP(32*4, 128, n=1, shortcut=False)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[-1]
        # x = self.conv(x)
        # x = self.spp(x)
        x = self.csp(x)
        return x
