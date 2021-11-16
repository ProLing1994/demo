'''
网络结构：tc-resnet14
平台：amba、novt
输入数据：2s 音频，图像输入长度为 196，宽度自适应
'''

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
    # 3s 音频建模
    def __init__(self, cfg):
        super().__init__()

        # init
        image_weidth = cfg.dataset.data_size[0]
        image_height = cfg.dataset.data_size[1]
        self.method = cfg.loss.method
        model_embedding_size = cfg.loss.embedding_size
        num_classes = cfg.dataset.label.num_classes

        self.planes = [16, 24, 32, 48, 64, 80]
        self.width_multiplier = 1.5
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
        self.layer5_1 = self._make_layer(TCBlock, int(self.planes[5] * self.width_multiplier), stride=2)
        self.layer5_2 = self._make_layer(TCBlock, int(self.planes[5] * self.width_multiplier), stride=1)

        self.conv2 = nn.Conv2d(int(self.planes[5] * self.width_multiplier), int(self.planes[5] * self.width_multiplier), kernel_size=(7, 1), stride=2, padding=(3, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(int(self.planes[5] * self.width_multiplier))

        if self.method == 'classification':
            self.conv3 = nn.Conv2d(int(self.planes[5] * self.width_multiplier), num_classes, kernel_size=(5, 1), stride=1, padding=(0, 0), bias=False)
        elif self.method == 'embedding':
            self.conv3 = nn.Conv2d(int(self.planes[5] * self.width_multiplier), model_embedding_size, kernel_size=(5, 1), stride=1, padding=(0, 0), bias=False)
        elif self.method == 'classification & embedding':
            self.conv3 = nn.Conv2d(int(self.planes[5] * self.width_multiplier), model_embedding_size, kernel_size=(5, 1), stride=1, padding=(0, 0), bias=False)
            self.conv4 = nn.Conv2d(model_embedding_size, num_classes, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=False)
        else:
            raise Exception("[Unknow:] cfg.loss.method. ")

    def _make_layer(self, block, planes, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1).contiguous()                          # shape: (batch, 1, 48, 296) ->  shape: (batch, 48, 296, 1) 
        x = x.permute(0, 3, 2, 1).contiguous()                          # shape: (batch, 1, 296, 48) ->  shape: (batch, 48, 296, 1) 
        out = self.relu(self.conv1(x))                                  # shape: (batch, 48, 296, 1) -> shape: (batch, 24, 296, 1)
        out = self.layer1_1(out)                                        # shape: (batch, 24, 296, 1) -> shape: (batch, 36, 148, 1)
        out = self.layer1_2(out)                                        # shape: (batch, 36, 148, 1) -> shape: (batch, 36, 148, 1)
        out = self.layer2_1(out)                                        # shape: (batch, 36, 148, 1) -> shape: (batch, 48, 74, 1)
        out = self.layer2_2(out)                                        # shape: (batch, 48, 74, 1) -> shape: (batch, 48, 74, 1)
        out = self.layer3_1(out)                                        # shape: (batch, 48, 74, 1) -> shape: (batch, 72, 37, 1)
        out = self.layer3_2(out)                                        # shape: (batch, 72, 37, 1) -> shape: (batch, 72, 37, 1)
        out = self.layer4_1(out)                                        # shape: (batch, 72, 37, 1) -> shape: (batch, 96, 19, 1)
        out = self.layer4_2(out)                                        # shape: (batch, 96, 19, 1) -> shape: (batch, 96, 19, 1)
        out = self.layer5_1(out)                                        # shape: (batch, 96, 19, 1) -> shape: (batch, 120, 10, 1)
        out = self.layer5_2(out)                                        # shape: (batch, 120, 10, 1) -> shape: (batch, 120, 10, 1)

        out = self.relu(self.bn2(self.conv2(out)))                      # shape: (batch, 120, 10, 1) -> shape: (batch, 120, 5, 1)
        if self.method == 'classification':
            out = self.conv3(out)                                           # shape: (batch, 120, 5, 1) -> shape: (batch, 2, 1, 1)
            return out
        elif self.method == 'embedding':
            embedding = self.conv3(out)                                           # shape: (batch, 120, 5, 1) -> shape: (batch, 128, 1, 1)
            return embedding
        elif self.method == 'classification & embedding':
            embedding = self.conv3(out)                                           # shape: (batch, 120, 5, 1) -> shape: (batch, 128, 1, 1)
            out = self.conv4(embedding)                                           # shape: (batch, 120, 1, 1) -> shape: (batch, 2, 1, 1)
            return embedding, out
        else:
            raise Exception("[Unknow:] cfg.loss.method. ")
        