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
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=94,
                          bias=True, kernel_size=(21, 8), 
                          stride=(1, 1))
    self.pool1 = nn.MaxPool2d(kernel_size=(2, 3))

    self.conv2 = nn.Conv2d(in_channels=94, out_channels=94,
                          bias=True, kernel_size=(6, 4), 
                          stride=(1, 1))
    self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))

    with torch.no_grad():
      x = Variable(torch.zeros(1, 1, image_height, image_weidth))
      x = self.pool1(self.conv1(x))
      x = self.pool2(self.conv2(x))
      conv_net_size = x.view(1, -1).size(1)
      last_size = conv_net_size
    
    self.lin = nn.Linear(conv_net_size, 32)
    self.dnn1 = nn.Linear(32, 128)
    self.dnn2 = nn.Linear(128, 128)
    self.output = nn.Linear(128, num_classes)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = F.relu(self.conv1(x)) # shape: (batch, 1, 101, 40) ->  shape: (batch, 94, 81, 33)
    x = self.dropout(x)
    x = self.pool1(x)         # shape: (batch, 94, 81, 33) ->  shape: (batch, 94, 40, 11)

    x = F.relu(self.conv2(x)) # shape: (batch, 94, 40, 11) ->  shape: (batch, 94, 35, 8)
    x = self.dropout(x)
    x = self.pool2(x)         # shape: (batch, 94, 35, 8) ->  shape: (batch, 94, 35, 8)
  
    x = x.view(x.size(0), -1) # shape: (batch, 26320)
    x = self.lin(x)           # shape: (batch, 26320) ->  shape: (batch, 32)

    x = self.dnn1(x)          # shape: (batch, 32) ->  shape: (batch, 128)
    x = F.relu(x)
    x = self.dropout(x)

    x = self.dnn2(x)          # shape: (batch, 128) ->  shape: (batch, 128)
    x = self.dropout(x)
    return self.output(x)     # shape: (batch, 12)