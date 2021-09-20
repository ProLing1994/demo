import sys
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS/demo')
from network.resnet_phoneme import *
import torch.nn as nn
import torch
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, resnet, num_class, with_fc=True):
        super(ResNet, self).__init__()
        self.with_fc=with_fc
        self.num_class=num_class
        self.resnet = resnet
        self.conv1=nn.Conv2d(128, 256, (5,5), stride=(1, 1),dilation=1, padding=(2, 2))
        self.conv2=nn.Conv2d(256, 256, (5,5), stride=(1, 1),dilation=1, padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        print('===with_fc=====', with_fc)
        if(self.with_fc):
            self.fc_out2=nn.Linear(1792, num_class, bias=True)
        else:
            self.conv_out2=nn.Conv2d(256, num_class, (3,16), stride=(1, 1),dilation=1, padding=(1, 0))
            
    def forward(self, x):
        res_out = self.resnet(x)
        x=self.conv1(res_out)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)  
        x=self.dropout(x)
        if(self.with_fc):
          trans1=torch.transpose(x, 1, 2)
          reshape=trans1.reshape(trans1.size()[0],trans1.size()[1],-1)
          fc=self.fc_out2(reshape)
        else:
          x=self.conv_out2(x)
          x=torch.transpose(x, 1, 2)
          fc=x.reshape(x.size()[0],x.size()[1],-1)
          
        fc=F.softmax(fc,dim=2)
        # fc=F.log_softmax(fc,dim=2)
        return fc

def ASR_English_Net(num_class, pretrained=False):
    res18 = resnet18(pretrained=pretrained)
    model = ResNet(res18, num_class=num_class, with_fc=False)
    # model = ResNet(res18, num_class=num_class, with_fc=True)
    return model

