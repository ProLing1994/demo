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
    self.lstm = nn.LSTM(40, 128, 2, dropout=0.2, batch_first=True)
    self.dnn1 = nn.Linear(128, 64)
    self.output = nn.Linear(64, num_classes, bias=True)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = x.reshape(x.shape[0], x.shape[2], x.shape[3])     # shape: (batch, 1, 301, 40) ->  shape: (batch, 301, 40)
    
    # lstm
    self.lstm.flatten_parameters()
    x, (ht, ct) = self.lstm(x)                # shape: (batch, 301, 40)  ->  shape: (batch, 301, 128)

    # dnn
    x = x.mean(1)                             # pooling, shape: (batch, 301, 128)  ->  shape: (batch, 128)
    x = self.dnn1(x)                          # shape: (batch, 128)  ->  shape: (batch, 64)
    x = self.dropout(x)
    return self.output(x)                     # shape: (batch, 2)