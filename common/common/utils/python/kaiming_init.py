import torch.nn as nn
import torch


def kaiming_weight_init(m, bn_std=0.02):

    classname = m.__class__.__name__
    if 'Conv2d' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm' in classname:
        if m.weight is not None:
            m.weight.data.normal_(1.0, bn_std)
            m.bias.data.zero_()
    elif 'Linear' in classname:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'LSTM' in classname:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param)

# TOTEST
def kaiming_initialize(m):
    """FUNCTION TO INITILIZE NETWORK PARAMETERS

    Arg:
        m (torch.nn.Module): torch nn module instance
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_normal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
        else:
            print("ERROR: " + name)

# TOTEST
def xavier_initialize(m):
    """FUNCTION TO INITILIZE NETWORK PARAMETERS

    Arg:
        m (torch.nn.Module): torch nn module instance
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
        else:
            print("ERROR: " + name)