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

class RESBlock(nn.Module):
    expansion = 1
    conv_kernel = (3, 3)
    conv_padding = (1, 1)

    def __init__(self, in_planes, planes, stride=1):
        super(RESBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=self.conv_kernel,
                               stride=stride, padding=self.conv_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=self.conv_kernel,
                               stride=1, padding=self.conv_padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.relu = torch.nn.ReLU()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.ReLU()
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SpeechResModel(nn.Module):
    # tc-resnet14: https://arxiv.org/abs/1904.03814
    def __init__(self, num_classes, image_height, image_weidth):
        super().__init__()

        # init
        num_features = 19 

        self.conv1 = nn.Conv2d(1, num_features, kernel_size=(7,7),
                               stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = torch.nn.ReLU() 

        self.layer1_1 = self._make_layer(RESBlock, num_features, num_features, stride=2)
        self.layer1_2 = self._make_layer(RESBlock, num_features, num_features, stride=1)
        self.layer2_1 = self._make_layer(RESBlock, num_features, num_features, stride=2)
        self.layer2_2 = self._make_layer(RESBlock, num_features, num_features, stride=1)
        self.layer3_1 = self._make_layer(RESBlock, num_features, num_features, stride=2)
        self.layer3_2 = self._make_layer(RESBlock, num_features, num_features, stride=1)
        self.layer4_1 = self._make_layer(RESBlock, num_features, num_features, stride=2)
        self.layer4_2 = self._make_layer(RESBlock, num_features, num_features, stride=1)

        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=(7, 1), stride=1, padding=(0, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features)

        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=(5, 1), stride=1, padding=(0, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(num_features)

        self.conv4 = nn.Conv2d(num_features, num_classes, kernel_size=(3, 3), stride=1, padding=(0, 0), bias=False)

    def _make_layer(self, block, in_planes, planes, stride):
        layers = []
        layers.append(block(in_planes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))                        # shape: (batch, 1, 201, 40) -> shape: (batch, 19, 201, 40)
        out = self.layer1_1(out)                                        # shape: (batch, 19, 201, 40) -> shape: (batch, 19, 101, 20)
        out = self.layer1_2(out)                                        # shape: (batch, 19, 101, 20) -> shape: (batch, 19, 101, 20)
        out = self.layer2_1(out)                                        # shape: (batch, 19, 101, 20) -> shape: (batch, 19, 50, 10)
        out = self.layer2_2(out)                                        # shape: (batch, 19, 51, 10) -> shape: (batch, 19, 51, 10)
        out = self.layer3_1(out)                                        # shape: (batch, 19, 51, 10) -> shape: (batch, 19, 26, 5)
        out = self.layer3_2(out)                                        # shape: (batch, 19, 26, 5) -> shape: (batch, 19, 26, 5)
        out = self.layer4_1(out)                                        # shape: (batch, 19, 26, 5) -> shape: (batch, 19, 13, 3)
        out = self.layer4_2(out)                                        # shape: (batch, 19, 13, 3) -> shape: (batch, 19, 13, 3)

        out = self.relu(self.bn2(self.conv2(out)))                      # shape: (batch, 19, 13, 3) -> shape: (batch, 19, 7, 3)
        out = self.relu(self.bn3(self.conv3(out)))                      # shape: (batch, 19, 7, 3) -> shape: (batch, 19, 3, 3)
        out = self.conv4(out)                                           # shape: (batch, 96, 3, 3) -> shape: (batch, 3, 1, 1)
        return out