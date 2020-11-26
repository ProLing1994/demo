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
                          bias=True, kernel_size=(30, 4), 
                          stride=(2, 2))
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                          bias=True, kernel_size=(30, 4), 
                          stride=(2, 2))

    with torch.no_grad():
      x = Variable(torch.zeros(1, 1, image_height, image_weidth))
      x = self.conv1(x)
      x = self.conv2(x)
      lstm_input_size = x.size(1) * x.size(3)
    
    self.lstm = nn.LSTM(lstm_input_size, 128, 2, dropout=0.2)
    self.dnn1 = nn.Linear(128, 64)
    self.output = nn.Linear(64, num_classes, bias=True)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # cnn
    x = F.relu(self.conv1(x))                 # shape: (batch, 1, 301, 40) ->  shape: (batch, 32, 136, 19)
    x = F.relu(self.conv2(x))                 # shape: (batch, 32, 136, 19) ->  shape: (batch, 64, 54, 8)
    x = x.permute(0, 2, 3, 1).contiguous()    # shape: (batch, 64, 54, 8) ->  shape: (batch, 54, 8, 64)
    b, t, _, _ = x.size()
    x = x.view(b, t, -1)                      # shape:(batch, 54, 8, 64) ->  shape: (batch, 54, 512)

    # lstm
    x = x.permute(1, 0, 2)                    # shape: (batch, 54, 512) ->  shape: (54, batch, 512) (t, b, f)
    x, (ht, ct) = self.lstm(x)                # shape: (54, batch, 512) ->  shape: (54, batch, 128)
    x = x.permute(1, 0, 2).contiguous()       # shape: (54, batch, 128) ->  shape: (batch, 54, 128) 

    # dnn
    x = x.mean(1)                             # pooling, shape: (batch, 54, 128)   ->  shape: (batch, 128) 
    x = self.dnn1(x)                          # shape: (batch, 128)  ->  shape: (batch, 64)
    x = self.dropout(x)
    return self.output(x)                     # shape: (batch, 2)