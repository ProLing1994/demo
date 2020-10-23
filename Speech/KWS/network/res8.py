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
    self.conv0 = nn.Conv2d(in_channels=1, out_channels=45,
                           kernel_size=(3, 3), padding=(1, 1), 
                           stride=(1, 1), bias=False)
    self.pool = nn.AvgPool2d(kernel_size=(4, 3))

    self.n_layers = 6
    self.convs = [nn.Conv2d(in_channels=45, out_channels=45,
                            kernel_size=(3, 3), padding=1, 
                            dilation=1, bias=False) for i in range(self.n_layers)]

    for i, conv in enumerate(self.convs):
        self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(45, affine=False))
        self.add_module("conv{}".format(i + 1), conv)

    self.output = nn.Linear(45, num_classes)

  def forward(self, x):
    for i in range(self.n_layers + 1):
        y = F.relu(getattr(self, "conv{}".format(i))(x))
        if i == 0:
            y = self.pool(y)
            old_x = y
        if i > 0 and i % 2 == 0:
            x = y + old_x
            old_x = x
        else:
            x = y
        if i > 0:
            x = getattr(self, "bn{}".format(i))(x)
  
    x = x.view(x.size(0), x.size(1), -1)     # shape: (batch, 45, 25, 13) ->  # shape: (batch, 45, 325)
    x = torch.mean(x, 2)      # shape: (batch, 45, 325) ->  # shape: (batch, 45)
    return self.output(x)     # shape: (batch, 12)