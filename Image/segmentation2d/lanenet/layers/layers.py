import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, mode='conv'):
        super(DownSample, self).__init__()
        if mode == 'conv':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 2, 2),
            )
        elif mode == 'maxpool':
            self.conv = nn.Sequential(
                nn.MaxPool2d(2, 2), 
                nn.Conv2d(in_channels, out_channels, 1, bias=False), 
            )
        elif mode == 'avgpool':
            self.conv = nn.Sequential(
                nn.AvgPool2d(2, 2), 
                nn.Conv2d(in_channels, out_channels, 1, bias=False), 
            )
        else:
            raise RuntimeError('unknown downsample mode')
        
        self.BN_ReLu = nn.Sequential(
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(), 
        )
        
    def forward(self, x):
        return self.BN_ReLu(self.conv(x))

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, mode='conv', factor=2):
        super(UpSample, self).__init__()
        if mode == 'conv':
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, factor, factor),
                # output = (input-1)stride + outputpadding - 2padding + kernelsize
            )
        elif mode == 'nearest':
            self.conv = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=factor), 
                nn.Conv2d(in_channels, out_channels, 1, bias=False), 
            )
        elif mode == 'bilinear':
            self.conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=factor), 
                nn.Conv2d(in_channels, out_channels, 1, bias=False), 
            )
        elif mode == 'ps':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*factor*factor, 1), 
                nn.PixelShuffle(factor), 
            )
        else:
            raise RuntimeError('unknown upsample mode')
        
        self.BN_ReLu = nn.Sequential(
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(), 
        )
        
    def forward(self, x):
        tmp = self.conv(x)
        return self.BN_ReLu(self.conv(x))

class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "glu":
        return F.glu
    if activation == "elu":
        return F.elu
    if activation == "celu":
        return F.celu
    if activation == "selu":
        return F.selu
    if activation == "relu6":
        return F.relu6
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError('activation function not support!')

def _get_norm_fn(norm):
    if norm == "BN":
        return nn.BatchNorm2d
    if norm == "GN":
        return nn.GroupNorm
    if norm == "LN":
        return nn.LayerNorm
    if norm == "IN":
        return nn.InstanceNorm2d
    if norm == "LRN":
        return nn.LocalResponseNorm
    raise RuntimeError('norm function not support!')


if __name__ == '__main__':
    
    x = torch.rand(8, 16, 32, 32)
    net1 = DownSample(16, 32, 'avgpool')
    net2 = UpSample(16, 8, 'bilinear')
    y1 = net1(x)
    y2 = net2(x)
    print('x  shape: ', x.shape)
    print('dw shape: ', y1.shape)
    print('up shape: ', y2.shape)
    
    
    