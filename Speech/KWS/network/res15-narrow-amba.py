import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.kaiming_init import kaiming_weight_init

def parameters_init(net):
  net.apply(kaiming_weight_init)

class SpeechResModel(nn.Module):
    def __init__(self, num_classes, image_height, image_weidth):
        super().__init__()
        num_features = 19
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=num_features,
                            kernel_size=(3, 3), padding=(1, 1), 
                            stride=(1, 1), bias=False)
        
        self.n_layers = 13
        self.convs = [nn.Conv2d(in_channels=num_features, out_channels=num_features,
                                kernel_size=(3, 3), padding=int(2**(i // 3)), 
                                dilation=int(2**(i // 3)), bias=False) for i in range(self.n_layers)]

        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(num_features, affine=False))
            self.add_module("conv{}".format(i + 1), conv)

        self.relu = torch.nn.ReLU() 
        self.output = nn.Linear(num_features, num_classes)
        # self.pool_final = nn.AvgPool2d(kernel_size=(image_height, image_weidth), stride=1, padding=0, ceil_mode=False, count_include_pad=True)
        # self.conv_final = nn.Conv2d(in_channels=num_features, out_channels=num_classes,
        #                        kernel_size=(1, 1), padding=(0, 0), 
        #                        stride=(1, 1), bias=False)=

    def forward(self, x):
        for i in range(self.n_layers + 1):
            y = self.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        
        x = x.view(x.size(0), x.size(1), -1)     # shape: (batch, 19, 201, 40) ->  # shape: (batch, 19, 8040)
        x = torch.mean(x, 2)      # shape: (batch, 19, 8040) ->  # shape: (batch, 19)
        x = self.output(x)
        return x
