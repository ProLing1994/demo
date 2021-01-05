import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def kaiming_weight_init(m, bn_std=0.02):
    classname = m.__class__.__name__
    if 'Conv2d' in classname:
        version_tokens = torch.__version__.split('.')
        if int(version_tokens[0]) == 0 and int(version_tokens[1]) < 4:
            nn.init.kaiming_normal(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        if m.weight:
            m.weight.data.normal_(1.0, bn_std)
            m.bias.data.zero_()
    elif 'Linear' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

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

    self.output = nn.Linear(num_features, num_classes)

  def forward(self, x):
    for i in range(self.n_layers + 1):
        y = F.relu(getattr(self, "conv{}".format(i))(x))
        if i == 0:
            old_x = y
        if i > 0 and i % 2 == 0:
            x = y + old_x
            old_x = x
        else:
            x = y
        if i > 0:
            x = getattr(self, "bn{}".format(i))(x)
  
    x = x.view(x.size(0), x.size(1), -1)     # shape: (batch, 19, 101, 40) ->  # shape: (batch, 19, 4040)
    x = torch.mean(x, 2)      # shape: (batch, 19, 4040) ->  # shape: (batch, 19)
    return self.output(x)     # shape: (batch, 12)