import numpy as np
import torch

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F


def define_loss_function(cfg):
    """ setup loss function
    :param cfg:
    :return:
    """
    if cfg.loss.name == 'softmax':
        loss_func = nn.CrossEntropyLoss()
    elif cfg.loss.name == 'focal':
        loss_func = FocalLoss(class_num=cfg.dataset.label.num_classes,
                              alpha=cfg.loss.obj_weight,
                              gamma=cfg.loss.focal_gamma)
    else:
        raise ValueError('Unsupported loss function.')
    return loss_func.cuda()


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.eye(class_num) / class_num
        else:
            assert len(alpha) == class_num
            self.alpha = torch.FloatTensor(alpha)
            self.alpha = self.alpha / self.alpha.sum()

        self.alpha = self.alpha.cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.one_hot_codes = torch.eye(self.class_num).cuda()

    def forward(self, input, target):
        # Assume that the input should has one of the following shapes:
        # 1. [sample, class_num]
        # 2. [batch, class_num, dim_y, dim_x]
        # 3. [batch, class_num, dim_z, dim_y, dim_x]
        assert input.dim() == 2 or input.dim() == 4 or input.dim() == 5
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input = input.view(input.numel() // self.class_num, self.class_num)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input = input.view(input.numel() // self.class_num, self.class_num)

        # Assume that the target should has one of the following shapes which
        # correspond to the shapes of the input:
        # 1. [sample, 1] or [sample, ]
        # 2. [batch, 1, dim_y, dim_x] or [batch, dim_y, dim_x]
        # 3. [batch, 1, dim_z, dim_y, dim_x], or [batch, dim_z, dim_y, dim_x]
        target = target.long().view(-1)

        # get alpha 
        alpha = self.alpha[target.data]
        alpha = Variable(alpha, requires_grad=False)

        mask = self.one_hot_codes[target.data]
        mask = Variable(mask, requires_grad=False)
        
        # softmax
        input = F.softmax(input, dim=1)

        # get probs from input
        probs = input * mask + (1 - input)*(1 - mask) + 1e-10
        log_probs = probs.log()

        if self.gamma > 0:
            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_probs
        else:
            batch_loss = -alpha * log_probs

        if self.size_average:
            # loss = batch_loss.mean()
            loss = batch_loss.sum()/mask.sum()
        else:
            loss = batch_loss.sum()
        return loss

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        # lam = np.random.uniform(0.2, 0.8)
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    floss = FocalLoss(4)
    input = torch.Tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
                          [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]).cuda()
    target = torch.Tensor([0, 1, 2, 1, 3]).cuda()
    input = Variable(input)
    target = Variable(target)
    floss(input, target)


if __name__ == "__main__":
    main()