from torch import nn
from torch.nn import functional as F


def loss_fn_kd(cfg, original_scores, teacher_scores, loss):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    assert cfg.knowledge_distillation.loss_name == 'kd'
    alpha = cfg.knowledge_distillation.alpha
    T = cfg.knowledge_distillation.temperature
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(original_scores/T, dim=1), F.softmax(teacher_scores/T, dim=1)) * (alpha * T * T) + \
                loss * (1. - alpha)

    return KD_loss


def loss_kl(original_scores, teacher_scores):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KL_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(original_scores, dim=1), F.softmax(teacher_scores, dim=1))
    return KL_loss