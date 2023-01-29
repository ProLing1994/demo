import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha=alpha
        self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        loss = self.alpha*loss
        return loss.mean()

class FocalLoss2d(nn.Module):
    '''
    input : b, c, w, h (outputs of the net) \n
    target: b, 1, w, h (label)
    '''
    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)
        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target, weight=weight)
        pt = torch.exp(logpt)
        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt
        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

if __name__ == '__main__':
    a = torch.Tensor([[1,0], [1,0], [0,1], [0,1]])
    b = torch.Tensor([0, 0, 1, 1])
    
    c = FocalLoss()
    d = c(a, b)
    print(d)
