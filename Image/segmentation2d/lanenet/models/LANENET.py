import torch
import torch.nn as nn
import torch.nn.functional as F

class InitialBlock_NOPOOL(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=0,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - As stated above the number of output channels for this
        # branch is the total minus 3, since the remaining channels come from
        # the extension branch
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x):
        main = self.main_branch(x)
        out = self.batch_norm(main)

        return self.out_prelu(out)

class RegularBottleneck(nn.Module):
    def __init__(self,
                 channels,
                 internal_ratio=2,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()


        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    bias=bias), nn.BatchNorm2d(internal_channels), activation,
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    bias=bias), nn.BatchNorm2d(internal_channels), activation)
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels))

        # PReLU layer to apply after adding the branches
        self.out_prelu = activation

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out)        

class DownsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=2,
                 kernel_size=3,
                 padding=0,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio < 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio
        #internal_channels = out_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            ceil_mode=True,
            return_indices=return_indices)

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels))
        
        # 1x1 expansion convolution
        self.ext_conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        main = self.ext_conv4(main)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        #ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out) #, max_indices


class LANENET_NOVT(nn.Module):
    def __init__(
            self,
            num_classes=5,
            sigmoid=False, 
    ):
        super(LANENET_NOVT, self).__init__()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.sigmoid = sigmoid

        self.initial_block_new = InitialBlock_NOPOOL(3, 16, padding=1, relu=True)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(16, 32, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=True)
        self.regular1_1 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=True)
        self.regular1_2 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=True)
        self.regular1_3 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=True)
        self.regular1_4 = RegularBottleneck(32, padding=1, dropout_prob=0.1, relu=True)
        

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(32, 64, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=True)
        self.regular2_1 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_2 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_3 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_4 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_5 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_6 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_7 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=True)     
        self.regular2_8 = RegularBottleneck(64,  padding=1, dropout_prob=0.1, relu=True)
        
        #deconv1
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        #deconv2
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU() 
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2, padding=0, bias=True),
        )
        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1]))
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2, 2]))

    
    def forward(self, img, seg_gt=None):
        x_0 = self.initial_block_new(img)

        x = self.downsample1_0(x_0)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x_1 = self.regular1_4(x)

        # Stage 2 - Encoder
        x = self.downsample2_0(x_1)
        x = self.regular2_1(x)
        x = self.regular2_2(x)
        x = self.regular2_3(x)
        x = self.regular2_4(x)
        x = self.regular2_5(x)
        x = self.regular2_6(x)
        x = self.regular2_7(x)
        x_2 = self.regular2_8(x)

        x = self.deconv1(x_2)
        deconv = self.deconv2(x)
        seg_pred = self.deconv3(deconv)
        
        if self.sigmoid:
            seg_pred = F.sigmoid(seg_pred)
        else:
            pass

        # if seg_gt is not None:
        #     loss = self.ce_loss(seg_pred, seg_gt)
        # else:
        #     loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred #, loss

# 20220913 修改 return_indices=False
class LANENET_BIG(nn.Module):
    def __init__(
            self,
            num_classes=5,
            sigmoid=False, 
            base_channel=32, 
    ):
        super(LANENET_BIG, self).__init__()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.sigmoid = sigmoid

        self.initial_block_new = InitialBlock_NOPOOL(3, base_channel, padding=1, relu=True)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(base_channel, base_channel*2, padding=1, return_indices=False, dropout_prob=0.1,
                                                    relu=True)
        self.regular1_1 = RegularBottleneck(base_channel*2, padding=1, dropout_prob=0.1, relu=True)
        self.regular1_2 = RegularBottleneck(base_channel*2, padding=1, dropout_prob=0.1, relu=True)
        self.regular1_3 = RegularBottleneck(base_channel*2, padding=1, dropout_prob=0.1, relu=True)
        self.regular1_4 = RegularBottleneck(base_channel*2, padding=1, dropout_prob=0.1, relu=True)
        

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(base_channel*2, base_channel*4, padding=1, return_indices=False, dropout_prob=0.1,
                                                    relu=True)
        self.regular2_1 = RegularBottleneck(base_channel*4,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_2 = RegularBottleneck(base_channel*4,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_3 = RegularBottleneck(base_channel*4,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_4 = RegularBottleneck(base_channel*4,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_5 = RegularBottleneck(base_channel*4,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_6 = RegularBottleneck(base_channel*4,  padding=1, dropout_prob=0.1, relu=True)
        self.regular2_7 = RegularBottleneck(base_channel*4,  padding=1, dropout_prob=0.1, relu=True)     
        self.regular2_8 = RegularBottleneck(base_channel*4,  padding=1, dropout_prob=0.1, relu=True)
        
        #deconv1
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(base_channel*4, base_channel*2, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(base_channel*2),
            nn.ReLU()
        )

        #deconv2
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(base_channel*2, base_channel*2, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(base_channel*2),
            nn.ReLU() 
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(base_channel*2, num_classes, kernel_size=2, stride=2, padding=0, bias=True),
        )
        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1]))
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2, 2]))

    
    def forward(self, img, seg_gt=None):
        x_0 = self.initial_block_new(img)

        x = self.downsample1_0(x_0)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x_1 = self.regular1_4(x)

        # Stage 2 - Encoder
        x = self.downsample2_0(x_1)
        x = self.regular2_1(x)
        x = self.regular2_2(x)
        x = self.regular2_3(x)
        x = self.regular2_4(x)
        x = self.regular2_5(x)
        x = self.regular2_6(x)
        x = self.regular2_7(x)
        x_2 = self.regular2_8(x)

        x = self.deconv1(x_2)
        deconv = self.deconv2(x)
        seg_pred = self.deconv3(deconv)
        
        if self.sigmoid:
            seg_pred = F.sigmoid(seg_pred)
        else:
            pass

        # if seg_gt is not None:
        #     loss = self.ce_loss(seg_pred, seg_gt)
        # else:
        #     loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred #, loss
    
    def forward_caffe(self, img):
        res = self.forward(img)
        res = F.softmax(res, 1)
        return res
    
if __name__ == '__main__':
    from torchsummary import summary
    
    net = LANENET_NOVT(4)
    
    x = torch.rand(8, 3, 16, 16)
    y = net(x)
    
    print(x.shape)
    print(y.shape)

    summary(net, (3,128,128), device='cpu')

