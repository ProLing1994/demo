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
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                          bias=True, kernel_size=(15, 32), 
                          stride=(3, 3))
    self.pool1 = nn.MaxPool2d(kernel_size=(1, 1))

    with torch.no_grad():
      x = Variable(torch.zeros(1, 1, image_height, image_weidth))
      x = self.pool1(self.conv1(x))
      conv_net_size = x.view(1, -1).size(1)

    self.dnn1 = nn.Linear(conv_net_size, 128, bias=True)
    self.dnn2 = nn.Linear(128, 128, bias=True)

    self.output = nn.Linear(128, num_classes, bias=True)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = F.relu(self.conv1(x)) # shape: (batch, 1, 101, 40) ->  shape: (batch, 186, 1, 33)
    x = self.dropout(x)
    x = self.pool1(x)         # shape: (batch, 186, 1, 33) ->  shape: (batch, 186, 1, 33)
  
    x = x.view(x.size(0), -1) # shape: (batch, 6138)
    x = self.dnn1(x)          # shape: (batch, 128)
    x = self.dropout(x)
    x = self.dnn2(x)          # shape: (batch, 128)
    x = self.dropout(x)
    return self.output(x)     # shape: (batch, 12)