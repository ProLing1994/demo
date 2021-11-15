
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def label_smoothing(inputs, num_classes=2, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 
    num_classes: num of classes
    epsilon: Smoothing rate. 
    '''
    return ((1 - epsilon) * inputs) + (epsilon / num_classes)


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, label_smoothing_on=False, label_smoothing_epsilon = 0.1):
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
        self.label_smoothing_on = label_smoothing_on
        self.label_smoothing_epsilon = label_smoothing_epsilon
        self.label_smoothing = label_smoothing(self.one_hot_codes, self.class_num, self.label_smoothing_epsilon)

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
        
        if self.label_smoothing_on:
            target_smoothing = self.label_smoothing[target.data]

        # softmax
        input = F.softmax(input, dim=1)

        # ''' 实现一： 正负类统一处理
        #     pt = pred * label + (1 - pred) * (1 - label)
        #     diff = (1-pt) ** self.gamma
        #     FocalLoss = -1 * alpha_t * diff * pt.log()
        #     代码简洁抽象，但不易兼容 Label Smooth
        #     [知乎](https://zhuanlan.zhihu.com/p/335694672)
        # '''
        # # get probs from input
        # pt = input * mask + (1 - input) * (1 - mask) + 1e-10
        # diff = (torch.pow((1 - pt), self.gamma))
        # log_pt = pt.log()

        # if self.gamma > 0:
        #     batch_loss = -alpha * diff * log_pt
        # else:
        #     batch_loss = -alpha * log_pt

        ''' 实现二： 正负类分别处理
            FocalLoss = - alpha_t * label * (1 - pred) ** self.gamma * torch.log(pred) - (1 - alpha_t) * (1 - label) * pred ** self.gamma * torch.log(1 - pred)
            代码稍复杂，但逻辑清晰，容易兼容 Label Smooth
            [知乎](https://zhuanlan.zhihu.com/p/335694672)
        '''
        log_pred = (input + 1e-10).log()
        log_pred_reverse = (1 - input + 1e-10).log()
        if self.label_smoothing_on:
            if self.gamma > 0:
                batch_loss = -alpha * (target_smoothing * (torch.pow((target_smoothing - input), self.gamma)) * log_pred + \
                                        (1 - target_smoothing) * (torch.pow((input - target_smoothing), self.gamma)) * log_pred_reverse)
            else:
                batch_loss = -alpha * (target_smoothing * log_pred + (1 - target_smoothing) * log_pred_reverse)
        else:
            if self.gamma > 0:
                batch_loss = -alpha * (mask * (torch.pow((1 - input), self.gamma)) * log_pred + \
                                        (1 - mask) * (torch.pow((input), self.gamma)) * log_pred_reverse)
            else:
                batch_loss = -alpha * (mask * log_pred + (1 - mask) * log_pred_reverse)

        if self.size_average:
            # loss = batch_loss.mean()
            loss = batch_loss.sum()/mask.sum()
        else:
            loss = batch_loss.sum()
        return loss