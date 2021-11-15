import sys
import torch
from torch import nn
from torch.autograd import Variable

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
from Basic.loss.ema import EMA
from Basic.loss.kd import loss_fn_kd, loss_kl
from Basic.loss.focal_loss import FocalLoss
from Basic.loss.amsoftmax import AMSoftmax


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
    elif cfg.loss.name == 'AmSoftmax':
        loss_func = AMSoftmax(in_feats = cfg.loss.embedding_size, 
                                n_classes = cfg.loss.num_classes,
                                m = cfg.loss.AmSoftmax_m,
                                s = cfg.loss.AmSoftmax_s)

    else:
        raise ValueError('Unsupported loss function.')
    return loss_func.cuda()


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