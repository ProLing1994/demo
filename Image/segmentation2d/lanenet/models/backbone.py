import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.vgg import *
from layers.rfb import *
from layers.resnet import *
from layers.layers import *
from layers.involution import *


class Seg32(nn.Module):
    '''
        input:  images (B, c, h, w)
        output: fetures(B, c, h, w)
        stream: 
        input-> (B,C,h,w)                 | conv0
         x0 --> (B,fc,h,w)                | conv1 ------------
         x1 --> (B,fc*2,h/2,w/2)          | conv2 ---------- |
         x2 --> (B,fc*4,h/4,w/4)          | conv3 -------- | |
         x3 --> (B,fc*8,h/8,w/8)          | conv4 ------ | | |
         x4 --> (B,fc*8,h/16,w/16)        | conv5 ---- | | | |
         x5 --> (B,fc*8,h/32,w/32)        | conv6 -- | | | | |
                                          |        | | | | | |
         x6 --> (B,fc*8,h/32,w/32)        | conv7 -- | | | | |
         x7 --> (B,fc*8,h/16,w/16)        | conv8 ---- | | | |
         x8 --> (B,fc*8,h/8,w/8)          | conv9 ------ | | |
         x9 --> (B,fc*4,h/4,w/4)          | conv10 ------- | |
         x10--> (B,fc*2,h/2,w/2)          | conv11 --------- |
         x11--> (B,fc,h,w)                | ------------------
    '''
    def __init__(self, img_c, img_hw, first_channel, block='RFBResBlock', nblock=1, 
                 downsample_mode='conv', upsample_mode='conv'):
        super(Seg32, self).__init__()
        h, w = img_hw
        fc = first_channel
        # input layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(img_c, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
            nn.Conv2d(fc, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
        )
        self.conv1 = nn.Sequential(
            xblock((fc, h, w), nblock, block), 
            DownSample(fc, fc*2, downsample_mode)
        )
        self.conv2 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            DownSample(fc*2, fc*4, downsample_mode)
        )
        self.conv3 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            DownSample(fc*4, fc*8, downsample_mode)
        )
        self.conv4 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            DownSample(fc*8, fc*8, downsample_mode)
        )
        self.conv5 = nn.Sequential(
            xblock((fc*8, h//16, w//16), nblock, block), 
            DownSample(fc*8, fc*8, downsample_mode)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(fc*8, fc*8, 3, 1, 1), 
            nn.BatchNorm2d(fc*8), 
            nn.LeakyReLU(0.1), 
        )
        self.conv7 = nn.Sequential(
            xblock((fc*8, h//32, w//32), nblock, block), 
            UpSample(fc*8, fc*8, upsample_mode)
        )
        self.conv8 = nn.Sequential(
            xblock((fc*8, h//16, w//16), nblock, block), 
            UpSample(fc*8, fc*8, upsample_mode)
        )
        self.conv9 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            UpSample(fc*8, fc*4, upsample_mode)
        )
        self.conv10 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            UpSample(fc*4, fc*2, upsample_mode)
        )
        self.conv11 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            UpSample(fc*2, fc, upsample_mode)
        )
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5) + x5
        
        x7 = self.conv7(x6) + x4
        x8 = self.conv8(x7) + x3
        x9 = self.conv9(x8) + x2
        x10 = self.conv10(x9) + x1
        x11 = self.conv11(x10) + x0
        return x11

class Seg16(nn.Module):
    '''
        input:  images (B, c, h, w)
        output: fetures(B, c, h, w)
        stream: 
        input-> (B,C,h,w)                 | conv0
         x0 --> (B,fc,h,w)                | conv1 ------------
         x1 --> (B,fc*2,h/2,w/2)          | conv2 ---------- |
         x2 --> (B,fc*4,h/4,w/4)          | conv3 -------- | |
         x3 --> (B,fc*8,h/8,w/8)          | conv4 ------ | | |
         x4 --> (B,fc*8,h/16,w/16)        | conv5      | | | |
         x8 --> (B,fc*8,h/8,w/8)          | conv9 ------ | | |
         x9 --> (B,fc*4,h/4,w/4)          | conv10 ------- | |
         x10--> (B,fc*2,h/2,w/2)          | conv11 --------- |
         x11--> (B,fc,h,w)                | ------------------
    '''
    def __init__(self, img_c, img_hw, first_channel, block='RFBResBlock', nblock=1, 
                 downsample_mode='conv', upsample_mode='conv'):
        super(Seg16, self).__init__()
        h, w = img_hw
        fc = first_channel
        # input layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(img_c, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.ReLU(), 
            nn.Conv2d(fc, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.ReLU(), 
        )
        self.conv1 = nn.Sequential(
            xblock((fc, h, w), nblock, block), 
            DownSample(fc, fc*2, downsample_mode)
        )
        self.conv2 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            DownSample(fc*2, fc*4, downsample_mode)
        )
        self.conv3 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            DownSample(fc*4, fc*8, downsample_mode)
        )
        self.conv4 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            DownSample(fc*8, fc*8, downsample_mode)
        )
        self.conv8 = nn.Sequential(
            xblock((fc*8, h//16, w//16), nblock, block), 
            UpSample(fc*8, fc*8, upsample_mode)
        )
        self.conv9 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            UpSample(fc*8, fc*4, upsample_mode)
        )
        self.conv10 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            UpSample(fc*4, fc*2, upsample_mode)
        )
        self.conv11 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            UpSample(fc*2, fc, upsample_mode)
        )
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x8 = self.conv8(x4) + x3
        x9 = self.conv9(x8) + x2
        x10 = self.conv10(x9) + x1
        x11 = self.conv11(x10) + x0
        return x11

class Fpn16(nn.Module):
    def __init__(self, img_c, img_hw, first_channel, block='RFBResBlock', nblock=1, 
                 downsample_mode='conv', upsample_mode='conv'):
        super(Fpn16, self).__init__()
        h, w = img_hw
        fc = first_channel
        # input layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(img_c, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
            nn.Conv2d(fc, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
        )
        self.conv1 = nn.Sequential(
            xblock((fc, h, w), nblock, block), 
            DownSample(fc, fc*2, downsample_mode)
        )
        self.conv2 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            DownSample(fc*2, fc*4, downsample_mode)
        )
        self.conv3 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            DownSample(fc*4, fc*8, downsample_mode)
        )
        self.conv4 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            DownSample(fc*8, fc*8, downsample_mode)
        )
        self.conv5 = xblock((fc*8, h//16, w//16), nblock, block)
        # up
        self.conv6_1x1 = nn.Conv2d(fc*8, fc*8, 1, bias=False)
        self.conv6_up = UpSample(fc*8, fc*8, mode=upsample_mode)
        self.conv6 = xblock((fc*8, h//8, w//8), nblock, block)
        
        self.conv7_1x1 = nn.Conv2d(fc*4, fc*4, 1, bias=False)
        self.conv7_up = UpSample(fc*8, fc*4, mode=upsample_mode)
        self.conv7 = xblock((fc*4, h//4, w//4), nblock, block)
        
        self.conv8_1x1 = nn.Conv2d(fc*2, fc*2, 1, bias=False)
        self.conv8_up = UpSample(fc*4, fc*2, mode=upsample_mode)
        self.conv8 =  xblock((fc*2, h//2, w//2), nblock, block)
        
        self.conv9_1x1 = nn.Conv2d(fc*1, fc*1, 1, bias=False)
        self.conv9_up = UpSample(fc*2, fc*1, mode=upsample_mode)
        self.conv9 = xblock((fc*1, h, w), nblock, block)
    
    def forward(self, x):
        x0 = self.conv0(x)  # ( fc, h, w)
        x1 = self.conv1(x0) # (2fc, h//2, w//2)
        x2 = self.conv2(x1) # (4fc, h//4, w//4)
        x3 = self.conv3(x2) # (8fc, h//8, w//8)
        x4 = self.conv4(x3) # (8fc, h//16, w//16)
        
        x5 = self.conv5(x4) # (8fc, h//16, w//16)
        x6 = self.conv6(self.conv6_1x1(x3) + self.conv6_up(x5)) # (8fc, h//8, w//8)
        x7 = self.conv7(self.conv7_1x1(x2) + self.conv7_up(x6)) # (4fc, h//4, w//4)
        x8 = self.conv8(self.conv8_1x1(x1) + self.conv8_up(x7)) # (2fc, h//2, w//2)
        x9 = self.conv9(self.conv9_1x1(x0) + self.conv9_up(x8)) # ( fc, h, w)
        return x9

class FCN16(nn.Module):
    def __init__(self, img_c, img_hw, first_channel, block='RFBResBlock', nblock=1, 
                 downsample_mode='conv', upsample_mode='conv'):
        super(FCN16, self).__init__()
        h, w = img_hw
        fc = first_channel
        # input layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(img_c, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
            nn.Conv2d(fc, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
        )
        self.conv1 = nn.Sequential(
            xblock((fc, h, w), nblock, block), 
            DownSample(fc, fc*2, downsample_mode)
        )
        self.conv2 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            DownSample(fc*2, fc*4, downsample_mode)
        )
        self.conv3 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            DownSample(fc*4, fc*8, downsample_mode)
        )
        self.conv4 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            DownSample(fc*8, fc*8, downsample_mode)
        )
        self.conv5 = xblock((fc, h, w), nblock, block)
        self.conv6 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            UpSample(fc*4, fc, 'ps', 4), 
        )
        self.conv7 = nn.Sequential(
            xblock((fc*8, h//16, w//16), nblock, block), 
            UpSample(fc*8, fc, 'ps', 16), 
        )
        self.convc = nn.Sequential(
            nn.Conv2d(fc*3, fc, 3, 1, 1), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
            nn.Conv2d(fc, fc, 3, 1, 1)
        )
        
    def forward(self, x):
        x0 = self.conv0(x)  # ( fc, h, w)
        x1 = self.conv1(x0) # (2fc, h//2, w//2)
        x2 = self.conv2(x1) # (4fc, h//4, w//4)
        x3 = self.conv3(x2) # (8fc, h//8, w//8)
        x4 = self.conv4(x3) # (8fc, h//16, w//16)
        
        x0 = self.conv5(x0) # ( fc, h, w)
        x2 = self.conv6(x2) # ( fc, h, w)
        x4 = self.conv7(x4) # ( fc, h, w)
        
        xc = torch.cat([x0, x2, x4], dim=1) # ( 3fc, h, w)
        xc = self.convc(xc) # ( fc, h, w)
        return xc

class mix16(nn.Module):
    def __init__(self, img_c, img_hw, first_channel, block='RFBResBlock', nblock=1, 
                 downsample_mode='conv', upsample_mode='conv'):
        super(mix16, self).__init__()
        h, w = img_hw
        fc = first_channel
        # input layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(img_c, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
            nn.Conv2d(fc, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
        )
        self.conv1 = nn.Sequential(
            xblock((fc, h, w), nblock, block), 
            DownSample(fc, fc, downsample_mode)
        )
        self.conv2 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            DownSample(fc*2, fc*2, downsample_mode)
        )
        self.conv3 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            DownSample(fc*4, fc*4, downsample_mode)
        )
        self.conv4 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            DownSample(fc*8, fc*4, downsample_mode)
        )
        self.conv1w = nn.Sequential(
            xblock((fc, h, w), 1, 'VggWide'), 
            DownSample(fc, fc, downsample_mode)
        )
        self.conv2w = nn.Sequential(
            xblock((fc*2, h//2, w//2), 1, 'VggWide'), 
            DownSample(fc*2, fc*2, downsample_mode)
        )
        self.conv3w = nn.Sequential(
            xblock((fc*4, h//4, w//4), 1, 'VggWide'), 
            DownSample(fc*4, fc*4, downsample_mode)
        )
        self.conv4w = nn.Sequential(
            xblock((fc*8, h//8, w//8), 1, 'VggWide'), 
            DownSample(fc*8, fc*4, downsample_mode)
        )
        self.conv8 = nn.Sequential(
            xblock((fc*8, h//16, w//16), nblock, block), 
            UpSample(fc*8, fc*4, upsample_mode)
        )
        self.conv9 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            UpSample(fc*8, fc*2, upsample_mode)
        )
        self.conv10 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            UpSample(fc*4, fc*1, upsample_mode)
        )
        self.conv11 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            UpSample(fc*2, fc//2, upsample_mode)
        )
        self.conv8w = nn.Sequential(
            xblock((fc*8, h//16, w//16), 1, 'VggWide'), 
            UpSample(fc*8, fc*4, upsample_mode)
        )
        self.conv9w = nn.Sequential(
            xblock((fc*8, h//8, w//8), 1, 'VggWide'), 
            UpSample(fc*8, fc*2, upsample_mode)
        )
        self.conv10w = nn.Sequential(
            xblock((fc*4, h//4, w//4), 1, 'VggWide'), 
            UpSample(fc*4, fc*1, upsample_mode)
        )
        self.conv11w = nn.Sequential(
            xblock((fc*2, h//2, w//2), 1, 'VggWide'), 
            UpSample(fc*2, fc//2, upsample_mode)
        )
        
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = torch.cat([self.conv1(x0), self.conv1w(x0)], 1)
        x2 = torch.cat([self.conv2(x1), self.conv2w(x1)], 1)
        x3 = torch.cat([self.conv3(x2), self.conv3w(x2)], 1)
        x4 = torch.cat([self.conv4(x3), self.conv4w(x3)], 1)
        x8 = torch.cat([self.conv8(x4), self.conv8w(x4)], 1) + x3
        x9 = torch.cat([self.conv9(x8), self.conv9w(x8)], 1) + x2
        x10 = torch.cat([self.conv10(x9), self.conv10w(x9)], 1) + x1
        x11 = torch.cat([self.conv11(x10), self.conv11w(x10)], 1) + x0
        return x11
        
class mix16s(nn.Module):
    def __init__(self, img_c, img_hw, first_channel, block='RFBResBlock', nblock=1, 
                 downsample_mode='conv', upsample_mode='conv'):
        super(mix16s, self).__init__()
        h, w = img_hw
        fc = first_channel
        # input layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(img_c, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
            nn.Conv2d(fc, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
        )
        self.conv1 = nn.Sequential(
            xblock((fc, h, w), nblock, block), 
            DownSample(fc, fc, downsample_mode)
        )
        self.conv2 = nn.Sequential(
            xblock((fc, h//2, w//2), nblock, block), 
            DownSample(fc, fc*2, downsample_mode)
        )
        self.conv3 = nn.Sequential(
            xblock((fc*2, h//4, w//4), nblock, block), 
            DownSample(fc*2, fc*4, downsample_mode)
        )
        self.conv4 = nn.Sequential(
            xblock((fc*4, h//8, w//8), nblock, block), 
            DownSample(fc*4, fc*4, downsample_mode)
        )
        self.conv1w = nn.Sequential(
            xblock((fc, h, w), 1, 'VggWide'), 
            DownSample(fc, fc, downsample_mode)
        )
        self.conv2w = nn.Sequential(
            xblock((fc, h//2, w//2), 1, 'VggWide'), 
            DownSample(fc, fc*2, downsample_mode)
        )
        self.conv3w = nn.Sequential(
            xblock((fc*2, h//4, w//4), 1, 'VggWide'), 
            DownSample(fc*2, fc*4, downsample_mode)
        )
        self.conv4w = nn.Sequential(
            xblock((fc*4, h//8, w//8), 1, 'VggWide'), 
            DownSample(fc*4, fc*4, downsample_mode)
        )
        self.conv8 = nn.Sequential(
            xblock((fc*8, h//16, w//16), nblock, block), 
            UpSample(fc*8, fc*8, upsample_mode)
        )
        self.conv9 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            UpSample(fc*8, fc*4, upsample_mode)
        )
        self.conv10 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            UpSample(fc*4, fc*2, upsample_mode)
        )
        self.conv11 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            UpSample(fc*2, fc, upsample_mode)
        )
        
    def forward(self, x):
        x0 = self.conv0(x)
        x1, x1w = self.conv1(x0), self.conv1w(x0)
        x2, x2w = self.conv2(x1), self.conv2w(x1w)
        x3, x3w = self.conv3(x2), self.conv3w(x2w)
        x4, x4w = self.conv4(x3), self.conv4w(x3w)
        x8 = self.conv8(torch.cat([x4, x4w], 1)) + torch.cat([x3, x3w], 1)
        x9 = self.conv9(x8) + torch.cat([x2, x2w], 1)
        x10 = self.conv10(x9) + torch.cat([x1, x1w], 1)
        x11 = self.conv11(x10) + x0
        return x11
        
class Seg16s(nn.Module):
    def __init__(self, img_c, img_hw, first_channel, block='RFBResBlock', nblock=1, 
                 downsample_mode='conv', upsample_mode='conv'):
        super(Seg16s, self).__init__()
        h, w = img_hw
        fc = first_channel
        # input layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(img_c, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
            nn.Conv2d(fc, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
        )
        self.conv1 = nn.Sequential(
            xblock((fc, h, w), nblock, block), 
            DownSample(fc, fc*2, downsample_mode)
        )
        self.conv2 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            DownSample(fc*2, fc*4, downsample_mode)
        )
        self.conv3 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            DownSample(fc*4, fc*8, downsample_mode)
        )
        self.conv4 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            DownSample(fc*8, fc*8, downsample_mode)
        )
        self.conv8 = nn.Sequential(
            xblock((fc*8, h//16, w//16), nblock, block), 
            UpSample(fc*8, fc*8, upsample_mode)
        )
        self.conv9 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            UpSample(fc*8, fc*4, upsample_mode)
        )
        self.conv10 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            UpSample(fc*4, fc*2, upsample_mode)
        )
        self.conv11 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            UpSample(fc*2, fc, upsample_mode)
        )
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x8 = self.conv8(x4) + x3
        x9 = self.conv9(x8) + x2
        x10 = self.conv10(x9) + x1
        x11 = self.conv11(x10) + x0
        return x11

class Seg8(nn.Module):
    '''
        input:  images (B, c, h, w)
        output: fetures(B, c, h, w)
        stream: 
        input-> (B,C,h,w)                 | conv0
         x0 --> (B,fc,h,w)                | conv1 ------------
         x1 --> (B,fc*2,h/2,w/2)          | conv2 ---------- |
         x2 --> (B,fc*4,h/4,w/4)          | conv3 -------- | |
         x3 --> (B,fc*8,h/8,w/8)          | conv4 ------ | | |
         x9 --> (B,fc*4,h/4,w/4)          | conv10 ------- | |
         x10--> (B,fc*2,h/2,w/2)          | conv11 --------- |
         x11--> (B,fc,h,w)                | ------------------
    '''
    def __init__(self, img_c, img_hw, first_channel, block='RFBResBlock', nblock=1, 
                 downsample_mode='conv', upsample_mode='conv'):
        super(Seg8, self).__init__()
        h, w = img_hw
        fc = first_channel
        # input layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(img_c, fc, 3, 1, 1), 
            nn.BatchNorm2d(fc), 
            nn.ReLU(), 
            nn.Conv2d(fc, fc, 3, 1, 1, groups=2), 
            nn.BatchNorm2d(fc), 
            nn.ReLU(), 
        )
        self.conv1 = nn.Sequential(
            xblock((fc, h, w), nblock, block), 
            DownSample(fc, fc*2, downsample_mode)
        )
        self.conv2 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            DownSample(fc*2, fc*4, downsample_mode)
        )
        self.conv3 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            DownSample(fc*4, fc*8, downsample_mode)
        )
        self.conv9 = nn.Sequential(
            xblock((fc*8, h//8, w//8), nblock, block), 
            UpSample(fc*8, fc*4, upsample_mode)
        )
        self.conv10 = nn.Sequential(
            xblock((fc*4, h//4, w//4), nblock, block), 
            UpSample(fc*4, fc*2, upsample_mode)
        )
        self.conv11 = nn.Sequential(
            xblock((fc*2, h//2, w//2), nblock, block), 
            UpSample(fc*2, fc, upsample_mode)
        )
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x9 = self.conv9(x3) + x2
        x10 = self.conv10(x9) + x1
        x11 = self.conv11(x10) + x0
        return x11

'''
class name(nn.Module):
    def __init__(self, img_c, img_hw, first_channel, block='RFBResBlock', nblock=1, 
                 downsample_mode='conv', upsample_mode='conv'):
        super(name, self).__init__()
        h, w = img_hw
        fc = first_channel
        # input layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(img_c, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
            nn.Conv2d(fc, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
        )
        self.conv1 = nn.Sequential(
            xblock((fc, h, w), nblock, block), 
            DownSample(fc, fc*2, downsample_mode)
        )
    def forward(self, x):
        return x
'''







class xblock(nn.Module):
    def __init__(self, shape, nblock, block):
        super(xblock, self).__init__()
        c, h, w = shape
        self.conv = nn.ModuleList()
        
        # mode | flops | params
        if block == 'VggBase': # 87M|226k
            BaseBlock = VggBase
        elif block == 'VggWide': # 160M|444k
            BaseBlock = VggWide
        elif block == 'ResBlockBase': # 129M|226k
            BaseBlock = ResBlockBase
        elif block == 'ResBlockBase2': # 129M|350k
            BaseBlock = ResBlockBase2
        elif block == 'ResBlockDense': # 129M|350k
            BaseBlock = ResBlockDense
        elif block == 'ResBlockEfficient': # 89M|230k
            BaseBlock = ResBlockEfficient
        elif block == 'RFBResBlock': # 120M|322k
            BaseBlock = RFBResBlock
        elif block == 'RFBResBlockPlus': # 130M|353k
            BaseBlock = RFBResBlockPlus
        elif block == 'RFBResBlockFree':
            BaseBlock = RFBResBlockFree
        elif block == 'involution':
            BaseBlock = invblock
        elif block == 'invres':
            BaseBlock = invResBlock
        else:
            raise RuntimeError('unknown block mode')
        
        for i in range(nblock):
            self.conv.append(BaseBlock(c, h, w))
                
    def forward(self, x):
        for m in self.conv:
            x = m(x)
        return x
