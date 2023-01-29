import torch.nn as nn
import torch
# from torchsummary import summary

class involution(nn.Module):

    def __init__(self, channels, h=None, w=None, stride=1, kernel_size=5, reduction_ratio=4, group_channels=8):
        super(involution, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels//reduction_ratio, kernel_size=1), 
            nn.BatchNorm2d(channels//reduction_ratio), 
            nn.ReLU6(), 
        )
        self.conv2 = nn.Conv2d(channels//reduction_ratio, kernel_size**2 * self.groups, kernel_size=1)

        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)
        
    def forward(self, x):
        
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)

        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        
        return out


class invblock(nn.Module):
    def __init__(self, c, *args):
        super(invblock, self).__init__()
        
        self.conv1 = involution(c)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), 
            nn.BatchNorm2d(c), 
            nn.ReLU6(), 
        )
        self.conv3 = nn.Conv2d(c*2, c, 1, bias=False)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        xc = torch.cat([x1, x2], 1)
        x = self.conv3(xc) + x
        return x
    
class invResBlock(nn.Module):
    def __init__(self, c, *args):
        super(invResBlock, self).__init__()
        
        self.conv = nn.Sequential(
            involution(c, kernel_size=5), 
            nn.BatchNorm2d(c), 
            nn.ReLU6(), 
            nn.Conv2d(c, c, 3, 1, 1), 
            nn.BatchNorm2d(c), 
            nn.ReLU6(), 
        )
        
    def forward(self, x):
        x = self.conv(x) + x
        return x


if __name__ == '__main__':
    
    c = 8
    # net = involution(channels=c, kernel_size=7, stride=1, reduction_ratio=1, group_channels=1)
    net = invResBlock(c)
    x = torch.rand(2, c, 64, 64)
    y = net(x)
    
    print(x.shape)
    print(y.shape)
    
    # summary(net, x.shape[1:], device='cpu')
    