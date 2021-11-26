import torch.nn as nn
import torch
import torch.nn.functional as F

from .resnet_asr import *

class ResNet(nn.Module):
    def __init__(self, resnet, num_class, with_fc=True):
        super(ResNet, self).__init__()
        self.with_fc=with_fc
        self.num_class=num_class
        self.resnet = resnet
        self.conv1s=nn.Conv2d(128, 256, (5,5), stride=(1, 1),dilation=1, padding=(2, 2))
        self.conv2s=nn.Conv2d(256, 128, (5,5), stride=(1, 1),dilation=1, padding=(2, 2))
        #self.conv1=nn.Conv2d(128, 256, (3,3), stride=(1, 1),dilation=1, padding=(0, 1))
        #self.conv2=nn.Conv2d(256, 128, (3,3), stride=(1, 1),dilation=1, padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        if(self.with_fc):
            self.pool=nn.MaxPool2d([1,12], stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            self.fc_out=nn.Linear(128, num_class,bias=True)
        else:
            self.conv_out=nn.Conv2d(128, num_class, (3,8), stride=(1, 1),dilation=1, padding=(0, 0))

    def forward(self, x):
        res_out = self.resnet(x)
        x=self.conv1s(res_out)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2s(x)
        x=self.bn2(x)
        x=self.relu(x)
        if(self.with_fc):
          x=self.pool(x)
          trans1=torch.transpose(x, 1, 2)
          reshape=trans1.reshape(trans1.size()[0],trans1.size()[1],-1)
          fc=self.fc_out(reshape)
        else:
          x=self.conv_out(x)
          x=torch.transpose(x, 1, 2)
          fc=x.reshape(x.size()[0],x.size()[1],-1)

        # fc=F.softmax(fc,dim=2)
        fc=F.log_softmax(fc,dim=2)
        return fc

def ASR_Mandarin_Net(num_class, pretrained=False):
    res18 = resnet18(pretrained=pretrained)
    model = ResNet(res18,num_class,with_fc=False)
    return model


