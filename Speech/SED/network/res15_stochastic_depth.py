import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.kaiming_init import kaiming_weight_init

def parameters_init(net):
    net.apply(kaiming_weight_init)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, prob, multFlag, in_planes, planes, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1]):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size[0], stride=stride[0], \
                                padding=padding[0], dilation=dilation[0], bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size[1], stride=stride[1], \
                                padding=padding[1], dilation=dilation[1], bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)

        self.relu = torch.nn.ReLU()
        self.shortcut = nn.Sequential()
        
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag

        # 经过处理后的 x 要与 x 的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积 +BN 来变换为同一维度
        if stride != [1, 1] or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, affine=False)
            )

    def forward(self, x):
        if self.training:
            if torch.equal(self.m.sample(),torch.ones(1)):
                # self.conv1.weight.requires_grad = True
                # self.conv2.weight.requires_grad = True

                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))

                out += self.shortcut(x)
                out = self.relu(out)
            else:
                # self.conv1.weight.requires_grad = False
                # self.conv2.weight.requires_grad = False

                out = self.shortcut(x)
                out = self.relu(out)
        else:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

            if self.multFlag:
                out = self.prob * out + self.shortcut(x)
                out = self.relu(out)
            else:
                out += self.shortcut(x)
                out = self.relu(out)
        return out

class SpeechResModel(nn.Module):
    def __init__(self, num_classes, image_height, image_weidth):
        super().__init__()

        del image_height, image_weidth

        # init 
        self.num_features = 45
        self.num_layers = 12
        self.in_planes = self.num_features
        self.padding_list = [int(2**(i // 3)) for i in range(self.num_layers + 1)]
        self.dilation_list = [int(2**(i // 3)) for i in range(self.num_layers + 1)]

        self.multFlag = True
        self.prob = [1, 0.5]
        self.prob_now = self.prob[0]
        self.prob_delta = self.prob[0] - self.prob[1]
        self.prob_step = self.prob_delta/(self.num_layers//2 - 1)

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=self.num_features, kernel_size=3, \
                                padding=1, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(self.num_features, affine=False)

        self.conv1 = nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=3, \
                                padding=self.padding_list[-1], stride=1, dilation=self.dilation_list[-1], bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_features, affine=False)
        self.relu = torch.nn.ReLU()

        self.layer1 = self._make_layer(BasicBlock, planes=self.num_features, kernel_size=[3, 3], stride=[1, 1], \
                                        padding=[self.padding_list[0], self.padding_list[1]], dilation=[self.dilation_list[0], self.dilation_list[1]])
        self.layer2 = self._make_layer(BasicBlock, planes=self.num_features, kernel_size=[3, 3], stride=[1, 1], \
                                        padding=[self.padding_list[2], self.padding_list[3]], dilation=[self.dilation_list[2], self.dilation_list[3]])
        self.layer3 = self._make_layer(BasicBlock, planes=self.num_features, kernel_size=[3, 3], stride=[1, 1], \
                                        padding=[self.padding_list[4], self.padding_list[5]], dilation=[self.dilation_list[4], self.dilation_list[5]])
        self.layer4 = self._make_layer(BasicBlock, planes=self.num_features, kernel_size=[3, 3], stride=[1, 1], \
                                        padding=[self.padding_list[6], self.padding_list[7]], dilation=[self.dilation_list[6], self.dilation_list[7]])
        self.layer5 = self._make_layer(BasicBlock, planes=self.num_features, kernel_size=[3, 3], stride=[1, 1], \
                                        padding=[self.padding_list[8], self.padding_list[9]], dilation=[self.dilation_list[8], self.dilation_list[9]])
        self.layer6 = self._make_layer(BasicBlock, planes=self.num_features, kernel_size=[3, 3], stride=[1, 1], \
                                        padding=[self.padding_list[10], self.padding_list[11]], dilation=[self.dilation_list[10], self.dilation_list[11]])

        self.output = nn.Linear(self.num_features, num_classes)

    def _make_layer(self, block, planes, kernel_size, stride, padding, dilation):
        layers = []
        layers.append(block(self.prob_now, self.multFlag, self.in_planes, planes, kernel_size, stride, padding, dilation))
        self.prob_now = self.prob_now - self.prob_step
        self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn0(self.conv0(x)))                        # shape: (batch, 1, 196, 64) -> shape: (batch, 45, 196, 64)
        out = self.layer1(out)                                          # shape: (batch, 45, 196, 64) -> shape: (batch, 45, 196, 64)
        out = self.layer2(out)                                          # shape: (batch, 45, 196, 64) -> shape: (batch, 45, 196, 64)
        out = self.layer3(out)                                          # shape: (batch, 45, 196, 64) -> shape: (batch, 45, 196, 64)
        out = self.layer4(out)                                          # shape: (batch, 45, 196, 64) -> shape: (batch, 45, 196, 64)
        out = self.layer5(out)                                          # shape: (batch, 45, 196, 64) -> shape: (batch, 45, 196, 64)
        out = self.layer6(out)                                          # shape: (batch, 45, 196, 64) -> shape: (batch, 45, 196, 64)
        out = self.relu(self.bn1(self.conv1(out)))                      # shape: (batch, 1, 196, 64) -> shape: (batch, 45, 196, 64)

        out = out.view(out.size(0), out.size(1), -1)                    # shape: (batch, 45, 196, 64) ->  # shape: (batch, 45, 12544)
        out = torch.mean(out, 2)                                        # shape: (batch, 45, 12544) ->  # shape: (batch, 45)
        out = self.output(out)                                          # shape: (batch, 45) ->  # shape: (batch, 2)
        return out