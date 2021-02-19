import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.kaiming_init import kaiming_weight_init

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from network.network_helper import draw_features


def parameters_init(net):
    net.apply(kaiming_weight_init)

class TCBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(TCBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(9, 1),
                               stride=stride, padding=(4, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(9, 1),
                               stride=1, padding=(4, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.relu = torch.nn.ReLU()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SpeechResModel(nn.Module):
    # tc-resnet8: https://arxiv.org/abs/1904.03814
    def __init__(self, num_classes, image_height, image_weidth):
        super().__init__()

        del image_height

        self.in_planes = 16

        self.conv1 = nn.Conv2d(image_weidth, self.in_planes, kernel_size=(3, 1),
                               stride=1, padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = torch.nn.ReLU() 

        self.layer1 = self._make_layer(TCBlock, 24, stride=2)
        self.layer2 = self._make_layer(TCBlock, 32, stride=2)
        self.layer3 = self._make_layer(TCBlock, 48, stride=2)
        self.linear = nn.Linear(48, num_classes)

    def _make_layer(self, block, planes, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1).contiguous()                          # shape: (batch, 1, 101, 40)  ->  shape: (batch, 40, 101, 1) 
        out = self.relu(self.bn1(self.conv1(x)))                        # shape: (batch, 40, 101, 1) -> shape: (batch, 16, 101, 1)
        out = self.layer1(out)                                          # shape: (batch, 16, 101, 1) -> shape: (batch, 24, 51, 1)
        out = self.layer2(out)                                          # shape: (batch, 24, 51, 1) -> shape: (batch, 32, 26, 1)
        out = self.layer3(out)                                          # shape: (batch, 32, 26, 1) -> shape: (batch, 48, 13, 1)
        out = out.view(out.size(0), out.size(1), -1)                    # shape: (batch, 48, 13, 1) ->  # shape: (batch, 48, 13)
        out = torch.mean(out, 2)                                        # shape: (batch, 48, 13) ->  # shape: (batch, 48)
        out = self.linear(out)
        return out