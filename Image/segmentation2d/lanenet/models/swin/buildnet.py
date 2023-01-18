import torch 
import torch.nn as nn
from models.swin.swin_transformer_ori import SwinTransformer




class RFBResBlock(nn.Module):
    def __init__(self, channel):
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
        return self.BN_ReLu(self.conv(x))


class decoder(nn.Module):
    def __init__(self, chs, num_classes):
        super(decoder, self).__init__()
        ch1, ch2, ch3, ch4 = chs
        self.up4 = nn.Sequential(
            UpSample(ch4, ch3), 
            RFBResBlock(ch3), 
            RFBResBlock(ch3), 
            RFBResBlock(ch3), 
        )
        self.up3 = nn.Sequential(
            UpSample(ch3, ch2), 
            RFBResBlock(ch2), 
            RFBResBlock(ch2), 
            RFBResBlock(ch2), 
        )
        self.up2 = nn.Sequential(
            UpSample(ch2, ch1), 
            RFBResBlock(ch1), 
            RFBResBlock(ch1), 
            RFBResBlock(ch1), 
        )
        self.up1 = nn.Sequential(
            UpSample(ch1, ch1, factor=4), 
            RFBResBlock(ch1), 
            RFBResBlock(ch1), 
            RFBResBlock(ch1), 
        )
        self.out = nn.Sequential(
            nn.Conv2d(ch1, num_classes, 3, 1, 1)
        )

    def forward(self, features):
        f1,f2,f3,f4 = features
        upf4 = self.up4(f4) + f3
        upf3 = self.up3(upf4) + f2
        upf2 = self.up2(upf3) + f1
        upf1 = self.up1(upf2)
        x = self.out(upf1)
        return x



class swin_seg(nn.Module):
    def __init__(self, imgsize, num_classes, sigmoid=False):
        super(swin_seg, self).__init__()
        self.do_sigmoid = sigmoid
        
        self.swin = SwinTransformer(imgsize)
        self.decode = decoder([96,192,384,768], num_classes)
    
    def forward(self, x):
        fs = self.swin(x)
        x = self.decode(fs)
        
        if self.do_sigmoid:
            x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    x = torch.rand(8, 3, 128, 128)
    net = swin_seg(imgsize=128, num_classes=5, sigmoid=True)

    y = net(x)
    print(y.shape)




