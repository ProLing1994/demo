import sys
import torch
from torch import Tensor
import torch.nn as nn

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.kaiming_init import kaiming_weight_init

def parameters_init(net):
    net.apply(kaiming_weight_init)


class SiLU(torch.nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

        
class SubSpectralNorm(nn.Module):
    def __init__(self, C, S, eps=1e-5):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = nn.BatchNorm2d(C*S)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)

        x = self.bn(x)

        return x.view(N, C, F, T)


class BroadcastedBlock(nn.Module):
    def __init__(
            self,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(BroadcastedBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation,
                                      stride=stride, bias=False)
        self.ssn1 = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.5)
        # self.swish = nn.SiLU()
        self.swish = SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)

        return out


class TransitionBlock(nn.Module):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(TransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride,
                                      dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.5)
        # self.swish = nn.SiLU()
        self.swish = SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # f2
        #############################
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.freq_dw_conv(out)
        out = self.ssn(out)
        #############################
        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        #############################

        out = auxilary + out
        out = self.relu(out)

        return out


class BCResNet(torch.nn.Module):
    def __init__(self, cfg=None):
        super(BCResNet, self).__init__()
        
        # init
        self.cfg = cfg
        self.num_classes = self.cfg.dataset.label.num_classes
        # self.width_multiplier = 1.0
        self.width_multiplier = 3.0

        self.conv1 = nn.Conv2d(1, int(16 * self.width_multiplier), 5, stride=(2, 1), padding=(2, 2))

        self.block1_1 = TransitionBlock(int(16 * self.width_multiplier), int(8* self.width_multiplier))
        self.block1_2 = BroadcastedBlock(int(8* self.width_multiplier))

        self.block2_1 = TransitionBlock(int(8* self.width_multiplier), int(12* self.width_multiplier), stride=(2, 1), dilation=(1, 2), temp_pad=(0, 2))
        self.block2_2 = BroadcastedBlock(int(12* self.width_multiplier), dilation=(1, 2), temp_pad=(0, 2))

        self.block3_1 = TransitionBlock(int(12* self.width_multiplier), int(16 * self.width_multiplier), stride=(2, 1), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_2 = BroadcastedBlock(int(16 * self.width_multiplier), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_3 = BroadcastedBlock(int(16 * self.width_multiplier), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_4 = BroadcastedBlock(int(16 * self.width_multiplier), dilation=(1, 4), temp_pad=(0, 4))

        self.block4_1 = TransitionBlock(int(16 * self.width_multiplier), int(20* self.width_multiplier), dilation=(1, 8), temp_pad=(0, 8))
        self.block4_2 = BroadcastedBlock(int(20* self.width_multiplier), dilation=(1, 8), temp_pad=(0, 8))
        self.block4_3 = BroadcastedBlock(int(20* self.width_multiplier), dilation=(1, 8), temp_pad=(0, 8))
        self.block4_4 = BroadcastedBlock(int(20* self.width_multiplier), dilation=(1, 8), temp_pad=(0, 8))

        self.conv2 = nn.Conv2d(int(20* self.width_multiplier), int(20* self.width_multiplier), 5, groups=int(20* self.width_multiplier), padding=(0, 2))
        self.conv3 = nn.Conv2d(int(20* self.width_multiplier), int(32* self.width_multiplier), 1, bias=False)
        self.conv4 = nn.Conv2d(int(32* self.width_multiplier), self.num_classes, 1, bias=False)

    def forward(self, x):

        x = x.permute(0, 1, 3, 2).contiguous()      # shape: (batch, 1, 101, 40) ->  shape: (batch, 1, 40, 101) 
        # print('INPUT SHAPE:', x.shape)
        out = self.conv1(x)                         # shape: (batch, 1, 101, 40) ->  shape: (batch, 16, 20, 101) 

        # print('BLOCK1 INPUT SHAPE:', out.shape)
        out = self.block1_1(out)                    # shape: (batch, 16, 20, 101) ->  shape: (batch, 8, 20, 101) 
        out = self.block1_2(out)                    # shape: (batch, 8, 20, 101) ->  shape: (batch, 8, 20, 101) 

        # print('BLOCK2 INPUT SHAPE:', out.shape)
        out = self.block2_1(out)                    # shape: (batch, 8, 20, 101) ->  shape: (batch, 12, 10, 101) 
        out = self.block2_2(out)                    # shape: (batch, 12, 20, 101) ->  shape: (batch, 12, 10, 101) 

        # print('BLOCK3 INPUT SHAPE:', out.shape)
        out = self.block3_1(out)                    # shape: (batch, 12, 20, 101) ->  shape: (batch, 16, 5, 101) 
        out = self.block3_2(out)                    # shape: (batch, 16, 5, 101) ->  shape: (batch, 16, 5, 101) 
        out = self.block3_3(out)                    # shape: (batch, 16, 5, 101) ->  shape: (batch, 16, 5, 101) 
        out = self.block3_4(out)                    # shape: (batch, 16, 5, 101) ->  shape: (batch, 16, 5, 101) 

        # print('BLOCK4 INPUT SHAPE:', out.shape)
        out = self.block4_1(out)                    # shape: (batch, 16, 5, 101) ->  shape: (batch, 20, 5, 101) 
        out = self.block4_2(out)                    # shape: (batch, 20, 5, 101) ->  shape: (batch, 20, 5, 101) 
        out = self.block4_3(out)                    # shape: (batch, 20, 5, 101) ->  shape: (batch, 20, 5, 101) 
        out = self.block4_4(out)                    # shape: (batch, 20, 5, 101) ->  shape: (batch, 20, 5, 101) 

        # print('Conv2 INPUT SHAPE:', out.shape)
        out = self.conv2(out)                       # shape: (batch, 20, 5, 101) ->  shape: (batch, 20, 1, 101) 

        # print('Conv3 INPUT SHAPE:', out.shape)
        out = self.conv3(out)                       # shape: (batch, 20, 1, 101) ->  shape: (batch, 32, 1, 101) 
        out = out.mean(-1, keepdim=True)            # shape: (batch, 32, 1, 101) ->  shape: (batch, 32, 1, 1) 

        # print('Conv4 INPUT SHAPE:', out.shape)
        out = self.conv4(out)                       # shape: (batch, 32, 1, 1) ->  shape: (batch, 12, 1, 1) 

        # print('OUTPUT SHAPE:', out.shape)
        return out

if __name__ == "__main__":
    x = torch.ones(5, 1, 40, 128)
    bcresnet = BCResNet()
    _ = bcresnet(x)
    print('num parameters:', sum(p.numel() for p in bcresnet.parameters() if p.requires_grad))