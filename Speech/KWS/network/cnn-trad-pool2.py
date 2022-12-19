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
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                          bias=True, kernel_size=(20, 8), 
                          stride=(1, 1))
    self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

    self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                          bias=True, kernel_size=(10, 4), 
                          stride=(1, 1))
    self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))

    with torch.no_grad():
      x = Variable(torch.zeros(1, 1, image_height, image_weidth))
      x = self.pool1(self.conv1(x))
      x = self.pool2(self.conv2(x))
      conv_net_size = x.view(1, -1).size(1)
      last_size = conv_net_size

    self.output = nn.Linear(last_size, num_classes)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = F.relu(self.conv1(x)) # shape: (batch, 1, 101, 40) ->  shape: (batch, 64, 82, 33)
    x = self.dropout(x)
    x = self.pool1(x)         # shape: (batch, 64, 82, 33) ->  shape: (batch, 64, 41, 16)

    x = F.relu(self.conv2(x)) # shape: (batch, 64, 41, 16) ->  shape: (batch, 64, 32, 13)
    x = self.dropout(x)
    x = self.pool2(x)
  
    x = x.view(x.size(0), -1) 
    return self.output(x)     # shape: (batch, 12)