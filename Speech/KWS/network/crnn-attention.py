import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.kaiming_init import kaiming_weight_init

def parameters_init(net):
    net.apply(kaiming_weight_init)
    nn.init.kaiming_normal_(net.weight_proj.data)
    nn.init.kaiming_normal_(net.weight_W.data)
    net.bias.data.zero_()

def batch_matmul_bias_like(seq, weight, bias, nonlinearity=''):
    s = torch.matmul(seq, weight)
    s += bias.squeeze(1).unsqueeze(0).expand_as(seq)
    if(nonlinearity=='tanh'):
        s = torch.tanh(s)
    return s.squeeze(dim=2)

def batch_matmul_like(seq, weight, nonlinearity=''):
    s = torch.matmul(seq, weight)
    if(nonlinearity=='tanh'):
        s = torch.tanh(s)
    return s.squeeze(dim=2)

def attention_mul_like(rnn_outputs, att_weights):
    attn_vectors = rnn_outputs * att_weights.unsqueeze(2).expand_as(rnn_outputs)
    return torch.sum(attn_vectors, 0)
    
class SpeechResModel(nn.Module):
  def __init__(self, num_classes, image_height, image_weidth, hidden_dim=128, num_layers=2):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                          bias=True, kernel_size=(20, 5), 
                          stride=(8, 2))

    with torch.no_grad():
      x = Variable(torch.zeros(1, 1, image_height, image_weidth))
      x = self.conv1(x)
      lstm_input_size = x.size(1) * x.size(3)
    
    # lstm
    self.lstm = nn.LSTM(lstm_input_size, hidden_dim, num_layers, dropout=0.2)

    # attention
    self.weight_proj = nn.Parameter(torch.Tensor(hidden_dim, 1))
    self.weight_W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
    self.bias = nn.Parameter(torch.Tensor(hidden_dim, 1))
    self.softmax = nn.Softmax(dim=1)

    # dnn
    self.dnn1 = nn.Linear(hidden_dim, 64)
    self.output = nn.Linear(64, num_classes, bias=True)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # cnn
    x = F.relu(self.conv1(x))                 # shape: (batch, 1, 301, 40) ->  shape: (batch, 32, 36, 18)
    x = x.permute(0, 2, 3, 1).contiguous()    # shape: (batch, 32, 36, 18) ->  shape: (batch, 36, 18, 32)
    b, t, _, _ = x.size()
    x = x.view(b, t, -1)                      # shape: (batch, 36, 18, 32) ->  shape: (batch, 36, 576)

    # lstm
    x = x.permute(1, 0, 2)                    # shape: (batch, 36, 576) ->  shape: (36, batch, 576) (t, b, f)
    self.lstm.flatten_parameters()
    x, (ht, ct) = self.lstm(x)                # shape: (36, batch, 576) ->  shape: (36, batch, 128)
    
    # attention
    squish = batch_matmul_bias_like(x, self.weight_W, self.bias, nonlinearity='tanh')        # shape: (36, batch, 128), tanh(self.weight_W * h_{t} + self.bias)
    attn = batch_matmul_like(squish, self.weight_proj)                                       # shape: (36, batch), e_{t} = self.weight_proj^T * tanh(self.weight_W * h_{t} + self.bias)
    attn_norm = self.softmax(attn.transpose(1,0))                                       # shape: (batch, 36), \alpha_{t} = softmax(e_{t})
    x = attention_mul_like(x, attn_norm.transpose(1,0))                                      # shape: (batch, 128), \sum_{t=1}^{T} \alpha_{t} h_{t}

    # dnn
    x = self.dnn1(x)                          # shape: (batch, 128)  ->  shape: (batch, 64)
    x = self.dropout(x)
    return self.output(x)                     # shape: (batch, 2)