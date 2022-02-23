import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
import torch.nn.init as init


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                      relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                      dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def conv_bn(inp, outp, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, outp, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(outp),
        nn.ReLU(inplace=True),
    )


def SeperableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels,
                  padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1)
    )


def conv_dw(inp, outp, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, outp, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outp),
        nn.ReLU(inplace=True),
    )


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        self.size = size

        # SSD network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.fpn = PyramidFeatures(128, 128, 128)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            detect4plate:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        fpn_sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(9):
            x = self.base[k](x)
            # print(x.shape)

        # s = self.L2Norm(x)
        # sources.append(x)
        fpn_sources.append(x)

        for k in range(9, 15):
            x = self.base[k](x)
        fpn_sources.append(x)

        # apply vgg up to fc7
        for k in range(15, len(self.base)):
            x = self.base[k](x)
            # print(x.shape)
        # sources.append(x)
        fpn_sources.append(x)

        features = self.fpn(fpn_sources)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            # print(x.shape)
            if k % 2 == 1:
                features.append(x)
                # print(x.shape)

        # apply multibox head to source layers
        for (x, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
base_channel = int(64 * 0.25)


def MobileNetV1(cfg, i, batch_norm=False):
    layers = []
    layers += [BasicConv(3, base_channel)]
    layers += [BasicConv(base_channel, base_channel, stride=2)]  # 150 * 150

    layers += [BasicConv(base_channel, base_channel * 2, stride=1)]
    layers += [BasicConv(base_channel * 2, base_channel * 2, stride=2)]  # 75 * 75

    layers += [BasicConv(base_channel * 2, base_channel * 4, stride=1)]
    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=1)]
    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=2)]  # 38 * 38

    layers += [BasicConv(base_channel * 4, base_channel * 8, stride=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=2)]  # 19 * 19

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1)]

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1)]

    layers += [BasicConv(base_channel * 8, 128, kernel_size=1, stride=1, padding=0)]
    layers += [BasicConv(128, 128, kernel_size=3, stride=2, padding=1)]  # 10*10

    return layers


base = {
    '300': ['S', 128, 128, 'S', 256, 256, 256, 'S', 512, 512, 512, 'S',
            512, 512, 512, 1024, 1024],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=128):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False

    layers += [BasicConv(128, 128, kernel_size=1, stride=1, padding=0)]
    layers += [BasicConv(128, 256, kernel_size=3, stride=2, padding=1)]  # 5 * 5

    layers += [BasicConv(256, 128, kernel_size=1, stride=1, padding=0)]
    layers += [BasicConv(128, 256, kernel_size=3, stride=1, padding=0)]  # 3 * 3

    layers += [BasicConv(256, 128, kernel_size=1, stride=1, padding=0)]
    layers += [BasicConv(128, 256, kernel_size=3, stride=1, padding=0)]  # 1 * 1

    return layers


extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256, 'S', 256],
}


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    # 38*38  512
    loc_layers += [nn.Conv2d(base_channel * 8, cfg[0] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(base_channel * 8, cfg[0] * num_classes, kernel_size=1, padding=0)]

    # 19*19  512
    loc_layers += [nn.Conv2d(base_channel * 8, cfg[1] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(base_channel * 8, cfg[1] * num_classes, kernel_size=1, padding=0)]

    # 10*10  256
    loc_layers += [nn.Conv2d(base_channel * 8, cfg[2] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(base_channel * 8, cfg[2] * num_classes, kernel_size=1, padding=0)]

    # 5*5  256
    loc_layers += [nn.Conv2d(256, cfg[3] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(256, cfg[3] * num_classes, kernel_size=1, padding=0)]

    # 3*3  256
    loc_layers += [nn.Conv2d(256, cfg[4] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(256, cfg[4] * num_classes, kernel_size=1, padding=0)]

    # 1*1  256
    loc_layers += [nn.Conv2d(256, cfg[5] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(256, cfg[5] * num_classes, kernel_size=1, padding=0)]

    return vgg, extra_layers, (loc_layers, conf_layers)


mbox = {
    '300': [6, 6, 6, 6, 6, 6],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    base_, extras_, head_ = multibox(MobileNetV1(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)


if __name__ == '__main__':
    # net = add_extras(extras[str(300)], 1024)
    net = MobileNetV1(base[str(300)], 3)
    nnnet = nn.Sequential(*net)
    # vgg_source = net[12][3].out_channels
    print(nnnet)
    '''
    from torchstat import stat
    net = build_net('test')
    stat(net,(3,300,300))
    '''
