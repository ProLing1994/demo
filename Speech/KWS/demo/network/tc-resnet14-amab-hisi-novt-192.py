import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class TCBlock(nn.Module):
    expansion = 1
    conv_kernel = (7, 1)
    conv_padding = (3, 0)

    def __init__(self, in_planes, planes, stride=1):
        super(TCBlock, self).__init__()
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
    def __init__(self, num_classes, image_height, image_weidth, width_multiplier=1.5):
        super().__init__()
        del image_height

        # init
        self.planes = [16, 24, 32, 48, 64]
        self.width_multiplier = width_multiplier
        self.first_conv_kernel = (3, 1)
        self.in_planes = int(self.planes[0] * self.width_multiplier)

        self.conv1 = nn.Conv2d(image_weidth, self.in_planes, kernel_size=self.first_conv_kernel,
                               stride=1, padding=(1, 0), bias=False)
        self.relu = torch.nn.ReLU() 

        self.layer1_1 = self._make_layer(TCBlock, int(self.planes[1] * self.width_multiplier), stride=2)
        self.layer1_2 = self._make_layer(TCBlock, int(self.planes[1] * self.width_multiplier), stride=1)
        self.layer2_1 = self._make_layer(TCBlock, int(self.planes[2] * self.width_multiplier), stride=2)
        self.layer2_2 = self._make_layer(TCBlock, int(self.planes[2] * self.width_multiplier), stride=1)
        self.layer3_1 = self._make_layer(TCBlock, int(self.planes[3] * self.width_multiplier), stride=2)
        self.layer3_2 = self._make_layer(TCBlock, int(self.planes[3] * self.width_multiplier), stride=1)
        self.layer4_1 = self._make_layer(TCBlock, int(self.planes[4] * self.width_multiplier), stride=2)
        self.layer4_2 = self._make_layer(TCBlock, int(self.planes[4] * self.width_multiplier), stride=1)

        self.conv2 = nn.Conv2d(int(self.planes[4] * self.width_multiplier), int(self.planes[4] * self.width_multiplier), kernel_size=(7, 1), stride=2, padding=(3, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(int(self.planes[4] * self.width_multiplier))

        self.conv3 = nn.Conv2d(int(self.planes[4] * self.width_multiplier), int(self.planes[4] * self.width_multiplier), kernel_size=(3, 1), stride=2, padding=(1, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(int(self.planes[4] * self.width_multiplier))

        self.conv4 = nn.Conv2d(int(self.planes[4] * self.width_multiplier), num_classes, kernel_size=(3, 1), stride=1, padding=(0, 0), bias=False)

    def _make_layer(self, block, planes, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1).contiguous()                          # shape: (batch, 1, 64, 192) ->  shape: (batch, 64, 192, 1) 
        x = x.permute(0, 3, 2, 1).contiguous()                          # shape: (batch, 1, 192, 64) ->  shape: (batch, 64, 192, 1) 
        out = self.relu(self.conv1(x))                                  # shape: (batch, 64, 192, 1) -> shape: (batch, 24, 192, 1)
        out = self.layer1_1(out)                                        # shape: (batch, 24, 192, 1) -> shape: (batch, 36, 96, 1)
        out = self.layer1_2(out)                                        # shape: (batch, 36, 96, 1) -> shape: (batch, 36, 96, 1)
        out = self.layer2_1(out)                                        # shape: (batch, 36, 96, 1) -> shape: (batch, 48, 48, 1)
        out = self.layer2_2(out)                                        # shape: (batch, 48, 48, 1) -> shape: (batch, 48, 48, 1)
        out = self.layer3_1(out)                                        # shape: (batch, 48, 48, 1) -> shape: (batch, 72, 24, 1)
        out = self.layer3_2(out)                                        # shape: (batch, 72, 24, 1) -> shape: (batch, 72, 24, 1)
        out = self.layer4_1(out)                                        # shape: (batch, 72, 24, 1) -> shape: (batch, 96, 12, 1)
        out = self.layer4_2(out)                                        # shape: (batch, 96, 12, 1) -> shape: (batch, 96, 12, 1)

        out = self.relu(self.bn2(self.conv2(out)))                      # shape: (batch, 96, 12, 1) -> shape: (batch, 96, 6, 1)
        out = self.relu(self.bn3(self.conv3(out)))                      # shape: (batch, 96, 6, 1) -> shape: (batch, 3, 3, 1)
        out = self.conv4(out)                                           # shape: (batch, 96, 3, 1) -> shape: (batch, 3, 1, 1)
        return out