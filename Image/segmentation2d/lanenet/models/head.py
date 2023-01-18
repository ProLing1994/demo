import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadCNN(nn.Module):
    def __init__(self, fc, nclass, softmax):
        super(HeadCNN, self).__init__()
        self.softmax = softmax
        self.head = nn.Sequential(
            nn.Conv2d(fc, fc, 5, 1, 2), 
            nn.BatchNorm2d(fc), 
            nn.LeakyReLU(0.1), 
            nn.Conv2d(fc, nclass, 5, 1, 2), 
        )
    def forward(self, x):
        x = self.head(x)
        if self.softmax:
            return F.softmax(x, dim=1)
        else:
            return x


