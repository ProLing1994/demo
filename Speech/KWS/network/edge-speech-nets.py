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


class SpeechResModel(nn.Module):
    # edge-speech-nets: https://arxiv.org/pdf/1810.08559.pdf
    def __init__(self, num_classes, image_height, image_weidth):
        super().__init__()

        del image_height
        del image_weidth
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=39,
                               kernel_size=(3, 3), padding=(1, 1),
                               stride=(1, 1), bias=False)

        self.conv1_1 = nn.Conv2d(in_channels=39, out_channels=20,
                               kernel_size=(3, 3), padding=(1, 1),
                               stride=(1, 1), bias=False)
        self.bn1_1 =  nn.BatchNorm2d(20, affine=False)
        self.conv1_2 = nn.Conv2d(in_channels=20, out_channels=39,
                            kernel_size=(3, 3), padding=(1, 1),
                            stride=(1, 1), bias=False)
        
        self.bn2_1 =  nn.BatchNorm2d(39, affine=False)
        self.conv2_1 = nn.Conv2d(in_channels=39, out_channels=15,
                               kernel_size=(3, 3), padding=(1, 1),
                               stride=(1, 1), bias=False)
        self.bn2_2 =  nn.BatchNorm2d(15, affine=False)
        self.conv2_2 = nn.Conv2d(in_channels=15, out_channels=39,
                            kernel_size=(3, 3), padding=(1, 1),
                            stride=(1, 1), bias=False)

        self.bn3_1 =  nn.BatchNorm2d(39, affine=False)
        self.conv3_1 = nn.Conv2d(in_channels=39, out_channels=25,
                               kernel_size=(3, 3), padding=(1, 1),
                               stride=(1, 1), bias=False)
        self.bn3_2 =  nn.BatchNorm2d(25, affine=False)
        self.conv3_2 = nn.Conv2d(in_channels=25, out_channels=39,
                            kernel_size=(3, 3), padding=(1, 1),
                            stride=(1, 1), bias=False)

        self.bn4_1 =  nn.BatchNorm2d(39, affine=False)
        self.conv4_1 = nn.Conv2d(in_channels=39, out_channels=22,
                               kernel_size=(3, 3), padding=(1, 1),
                               stride=(1, 1), bias=False)
        self.bn4_2 =  nn.BatchNorm2d(22, affine=False)
        self.conv4_2 = nn.Conv2d(in_channels=22, out_channels=39,
                            kernel_size=(3, 3), padding=(1, 1),
                            stride=(1, 1), bias=False)

        self.bn5_1 =  nn.BatchNorm2d(39, affine=False)
        self.conv5_1 = nn.Conv2d(in_channels=39, out_channels=22,
                               kernel_size=(3, 3), padding=(1, 1),
                               stride=(1, 1), bias=False)
        self.bn5_2 =  nn.BatchNorm2d(22, affine=False)
        self.conv5_2 = nn.Conv2d(in_channels=22, out_channels=39,
                            kernel_size=(3, 3), padding=(1, 1),
                            stride=(1, 1), bias=False)

        self.bn6_1 =  nn.BatchNorm2d(39, affine=False)
        self.conv6_1 = nn.Conv2d(in_channels=39, out_channels=25,
                               kernel_size=(3, 3), padding=(1, 1),
                               stride=(1, 1), bias=False)
        self.bn6_2 =  nn.BatchNorm2d(25, affine=False)
        self.conv6_2 = nn.Conv2d(in_channels=25, out_channels=39,
                            kernel_size=(3, 3), padding=(1, 1),
                            stride=(1, 1), bias=False)

        self.bn7_1 =  nn.BatchNorm2d(39, affine=False)
        self.conv7_1 = nn.Conv2d(in_channels=39, out_channels=45,
                               kernel_size=(3, 3), padding=(1, 1),
                               stride=(1, 1), bias=False)
        self.bn7_2 =  nn.BatchNorm2d(45, affine=False)

        self.relu = torch.nn.ReLU()
        self.output = nn.Linear(45, num_classes)

    def forward(self, x):
        x = self.relu(self.conv0(x))                                    # shape: (batch, 1, 101, 40) -> shape: (batch, 39, 101, 40)
        
        res_block = self.bn1_1(self.relu(self.conv1_1(x)))
        res_block = self.relu(self.conv1_2(res_block))
        x = x + res_block                                               # shape: (batch, 39, 101, 40)

        res_block = self.bn2_1(x)
        res_block = self.bn2_2(self.relu(self.conv2_1(res_block)))
        res_block = self.relu(self.conv2_2(res_block))
        x = x + res_block                                               # shape: (batch, 39, 101, 40)

        res_block = self.bn3_1(x)
        res_block = self.bn3_2(self.relu(self.conv3_1(res_block)))
        res_block = self.relu(self.conv3_2(res_block))
        x = x + res_block                                               # shape: (batch, 39, 101, 40)

        res_block = self.bn4_1(x)
        res_block = self.bn4_2(self.relu(self.conv4_1(res_block)))
        res_block = self.relu(self.conv4_2(res_block))
        x = x + res_block                                               # shape: (batch, 39, 101, 40)

        res_block = self.bn5_1(x)
        res_block = self.bn5_2(self.relu(self.conv5_1(res_block)))
        res_block = self.relu(self.conv5_2(res_block))
        x = x + res_block                                               # shape: (batch, 39, 101, 40)

        res_block = self.bn6_1(x)
        res_block = self.bn6_2(self.relu(self.conv6_1(res_block)))
        res_block = self.relu(self.conv6_2(res_block))
        x = x + res_block                                               # shape: (batch, 39, 101, 40)

        x = self.bn7_1(x)
        x = self.bn7_2(self.relu(self.conv7_1(x)))                      # shape: (batch, 39, 101, 40)

        x = x.view(x.size(0), x.size(1), -1)                            # shape: (batch, 45, 101, 40) -> shape: (batch, 45, 4040)
        x = torch.mean(x, 2)                                            # shape: (batch, 45, 4040) -> shape: (batch, 45)
        return self.output(x)                                           # shape: (batch, 45) -> shape: (batch, 12)
