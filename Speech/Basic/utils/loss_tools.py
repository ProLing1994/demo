import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

def loss_function(cfg):
    """ setup loss function
    :param cfg:
    :return:
    """
    if cfg.loss.name == 'softmax':
        loss_func = nn.CrossEntropyLoss()
    elif cfg.loss.name == 'focal':
        loss_func = FocalLoss(class_num=cfg.loss.num_classes,
                              alpha=cfg.loss.obj_weight,
                              gamma=cfg.loss.focal_gamma,
                              label_smoothing_on = cfg.regularization.label_smoothing.on,
                              label_smoothing_epsilon = cfg.regularization.label_smoothing.epsilon)
    else:
        raise ValueError('Unsupported loss function.')
    return loss_func.cuda()


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


class EMA(object):
    '''
        apply expontential moving average to a model. This should have same function as the `tf.train.ExponentialMovingAverage` of tensorflow.
        usage:
            model = resnet()
            model.train()
            ema = EMA(model, 0.9999)
            ....
            for img, lb in dataloader:
                loss = ...
                loss.backward()
                optim.step()
                ema.update_params() # apply ema
            evaluate(model)  # evaluate with original model as usual
            ema.apply_shadow() # copy ema status to the model
            evaluate(model) # evaluate the model with ema paramters
            ema.restore() # resume the model parameters

        args:
            - model: the model that ema is applied
            - alpha: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
            - buffer_ema: whether the model buffers should be computed with ema method or just get kept
        methods:
            - update_params(): apply ema to the model, usually call after the optimizer.step() is called
            - apply_shadow(): copy the ema processed parameters to the model
            - restore(): restore the original model parameters, this would cancel the operation of apply_shadow()
    '''
    def __init__(self, model, alpha, buffer_ema=True):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = self.model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name]
                + (1 - decay) * state[name]
            )
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }


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