import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.kaiming_init import kaiming_weight_init


def parameters_init(net):
    net.apply(kaiming_weight_init)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn14(nn.Module):
    def __init__(self, num_classes, image_height, image_weidth):
        super().__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x):

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')      # shape: (batch, 1, 501, 64) ->  # shape: (batch, 64, 250, 32)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')      # shape: (batch, 64, 250, 32) ->  # shape: (batch, 128, 125, 16)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')      # shape: (batch, 128, 125, 16) ->  # shape: (batch, 256, 62, 8)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')      # shape: (batch, 256, 62, 8) ->  # shape: (batch, 512, 31, 4)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')      # shape: (batch, 512, 31, 4) ->  # shape: (batch, 1024, 15, 2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')      # shape: (batch, 1024, 15, 2) ->  # shape: (batch, 2048, 15, 2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)                                        # shape: (batch, 2048, 15, 2)->  # shape: (batch, 2048, 15)
        
        (x1, _) = torch.max(x, dim=2)                                   # shape: (batch, 2048, 15)->  # shape: (batch, 2048)
        x2 = torch.mean(x, dim=2)                                       # shape: (batch, 2048, 15)->  # shape: (batch, 2048)
        x = x1 + x2                                                     # shape: (batch, 2048)->  # shape: (batch, 2048)
        x = F.dropout(x, p=0.5, training=self.training)                 # shape: (batch, 2048)->  # shape: (batch, 2048)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        output = self.fc_audioset(x)                                    # shape: (batch, 2048)->  # shape: (batch, 50)
        return output
