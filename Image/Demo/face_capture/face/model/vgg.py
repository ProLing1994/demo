
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmcv.cnn import ConvModule, xavier_init, caffe2_xavier_init
from mmcv.ops import DeformConv2dPack

import sys
sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Demo.face_capture.face.model.spp_neck import SPP_Neck


class FPNDcnLconv3Dcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPNDcnLconv3Dcn, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins # 5
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs: 
            if extra_convs_on_inputs:
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level): # (0, 5)
            l_conv = DeformConv2dPack(
                in_channels[i],
                out_channels,
                3,
                padding=1
                )
            fpn_conv = DeformConv2dPack(
                (self.backbone_end_level - i) * out_channels,
                out_channels,
                3,
                padding=1
                )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    #@auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ] # 改变通道数，无下采样

        # build top-down path
        used_backbone_levels = len(laterals) # 5
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = torch.cat((laterals[i - 1], F.interpolate(laterals[i], **self.upsample_cfg)), 1) 
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = torch.cat((laterals[i - 1], F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)), 1)

        if self.training:
            outs = [
                self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            ]
        else:   
            outs = [
                self.fpn_convs[0](laterals[0])
            ]
        
        return tuple(outs)


class HRFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False,
                 stride=1):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.reduction_conv = ConvModule(
            sum(in_channels),
            out_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            act_cfg=None)

        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    act_cfg=None))

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_ins
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            outs.append(
                F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        if out.requires_grad and self.with_cp:
            out = checkpoint(self.reduction_conv, out)
        else:
            out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        outputs = []

        for i in range(self.num_outs):
            if outs[i].requires_grad and self.with_cp:
                tmp_out = checkpoint(self.fpn_convs[i], outs[i])
            else:
                tmp_out = self.fpn_convs[i](outs[i])
            outputs.append(tmp_out)
        return tuple(outputs)


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 128,
                 mid_channels: int = 32,
                 dilation: int = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale  ## 1
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

base_channel = int(64 * 0.25)  ## 16
def VGG_RFB():
    layers = []
    layers += [BasicConv(3, base_channel, 1, 1, 0)]
    layers += [BasicConv(base_channel, base_channel, kernel_size=3, stride=2, padding=1)] #150 * 150

    layers += [BasicConv(base_channel, base_channel * 2, kernel_size=3,stride=1, padding=1)]
    layers += [BasicConv(base_channel * 2, base_channel * 2, stride=2, kernel_size=3, padding=1)] #75 * 75

    layers += [BasicConv(base_channel * 2, base_channel * 4, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=2, kernel_size=3, padding=1)] #38 * 38

    layers += [BasicConv(base_channel * 4, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicRFB(base_channel * 8, base_channel * 8, stride = 1, scale=1.0)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=2, kernel_size=3, padding=1)] # 19 * 19

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]

    layers += [BasicConv(base_channel * 8, 128, kernel_size=1, stride=1, padding=0)]
    layers += [BasicConv(128, 128, kernel_size=3, stride=2, padding=1)] # 10*10

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=2, kernel_size=3, padding=1)]

    return layers

BN_MOMENTUM = 0.1
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



class PoseVggNet(nn.Module):
    def __init__(self, heads, head_conv):
        super(PoseVggNet, self).__init__()
        self.inplanes = 128
        self.heads = heads 
        self.deconv_with_bias = False
        vgg = VGG_RFB()
        self.base = nn.ModuleList(vgg)
        # self.deconv_layers = self._make_deconv_layer(
        #     3,
        #     [128, 128, 128],
        #     [4, 4, 4],
        # )

        self.up0 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.up1 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, 2, 0)

        self.conv0 = nn.Conv2d(128, 128, 1)
        self.conv1 = nn.Conv2d(128, 128, 1)
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.conv3 = nn.Conv2d(32, 128, 1)

        # self.block_dilations = [2, 4, 6, 8]

        # self.fpn_dcn_lconv3_dcn = FPNDcnLconv3Dcn([32,64, 128, 128, 128], 128, 5)

        # self.hrfpn = HRFPN([32,64, 128, 128, 128], 128, 5)

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:  ## 128
                fc = nn.Sequential(
                  nn.Conv2d(128, classes,
                    kernel_size=1, stride=1,
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes,
                  kernel_size=1, stride=1,
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)
        self.spp_neck = SPP_Neck()
        # self._init_layers()
        self.init_weights()

    def _init_layers(self):
        encoder_blocks = []
        for i in range(4):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    128,
                    32,
                    dilation=dilation
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding


    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):

            kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            # fc = DCN(self.inplanes, planes,
            #             kernel_size=(3,3), stride=1,
            #             padding=1, dilation=1, deformable_groups=1)
            fc = nn.Conv2d(self.inplanes, planes,
                        kernel_size=3, stride=1,
                        padding=3, dilation=3, bias=False)
                # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                        in_channels=planes,
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


    def init_weights(self):
        for m in self.spp_neck.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 

    def forward(self, x):
        y = []
        for i in range(22):
            x = self.base[i](x)
            if i == 3:
                x1 = x
                y.append(x1)
            if i == 6:
                x2 = x
                y.append(x2)
            if i == 11:
                x3 = x
                y.append(x3)
            if i == 18:
                x4 = x
                y.append(x4)
            if i == 21:
                x5 = x
                y.append(x5)
        
        x4 = self.conv0(x4) + self.up0(x5)
        x3 = self.conv1(x3) + self.up1(x4)
        x2 = self.conv2(x2) + self.up2(x3)
        x = self.conv3(x1) + self.up3(x2)

        # YOLOF Neck       
        # # x = self.dilated_encoder_blocks(x)  

        # SPP Neck
        x = self.spp_neck(x)

        # FPNDcnLconv3Dcn Neck
        # y1 = self.fpn_dcn_lconv3_dcn(y)
        # y2 = y1[0]
       
        # HRFPN Neck
        # y1 = self.hrfpn(y)
        # y2 = y1[0]

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        # for key, value in ret.items():
        #     print(value.shape)
        # sys.exit()
        
        return [ret]


class PoseVggNet_MOT_AlignPS(nn.Module):
    def __init__(self, heads, head_conv):
        super(PoseVggNet_MOT_AlignPS, self).__init__()
        self.inplanes = 128
        self.heads = heads  
        self.deconv_with_bias = False
        vgg = VGG_RFB()
        self.base = nn.ModuleList(vgg)
        # self.deconv_layers = self._make_deconv_layer(
        #     3,
        #     [128, 128, 128],
        #     [4, 4, 4],
        # )

        self.up0 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.up1 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, 2, 0)

        self.conv0 = nn.Conv2d(128, 128, 1)
        self.conv1 = nn.Conv2d(128, 128, 1)
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.conv3 = nn.Conv2d(32, 128, 1)
        self.block_dilations = [2, 4, 6, 8]

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:  ## 128
                fc = nn.Sequential(
                  nn.Conv2d(128, classes,
                    kernel_size=1, stride=1,
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes,
                  kernel_size=1, stride=1,
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)
        self.spp_neck = SPP_Neck()
        self.init_weights()

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding


    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):

            kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
    
            fc = nn.Conv2d(self.inplanes, planes,
                        kernel_size=3, stride=1,
                        padding=3, dilation=3, bias=False)
               
            up = nn.ConvTranspose2d(
                        in_channels=planes,
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


    def init_weights(self):
        if 1:
            for m in self.spp_neck.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0) 

    def forward(self, x):

        for i in range(22):
            x = self.base[i](x)
            if i == 3:
                x1 = x
            if i == 6:
                x2 = x
            if i == 11:
                x3 = x
            if i == 18:
                x4 = x
            if i == 21:
                x5 = x
       
        x4 = self.conv0(x4) + self.up0(x5)
        x3 = self.conv1(x3) + self.up1(x4)
        x2 = self.conv2(x2) + self.up2(x3)
        x = self.conv3(x1) + self.up3(x2)

        # YOLOF Neck       
        # x = self.dilated_encoder_blocks(x)  

        # SPP Neck
        x = self.spp_neck(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        
        return [ret]


def get_vggnet():
    # vgg net 
    model = PoseVggNet(heads={'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}, head_conv=128)
    return model