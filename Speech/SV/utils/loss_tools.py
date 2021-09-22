import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def ge2e_loss(embeds, sim_matrix, loss_fn):
    """
    Computes the softmax loss according to GE2E.
    
    :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
    utterances_per_speaker, embedding_size)
    :param loss_fn: loss function
    :return: the loss and the EER for this batch of embeddings.
    """
    speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

    # sim_matrix
    sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, speakers_per_batch))

    # target
    ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
    target = torch.from_numpy(ground_truth).long().cuda()

    # Loss
    loss = loss_fn(sim_matrix, target)
    
    # EER (not backpropagated)
    with torch.no_grad():
        inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
        labels = np.array([inv_argmax(i) for i in ground_truth])
        preds = sim_matrix.detach().cpu().numpy()

        # Snippet from https://yangcha.github.io/EER-ROC/
        fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
    return loss, eer


def compute_eer(embeds, sim_matrix):
    """
    Computes EER.
    
    :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
    utterances_per_speaker, embedding_size)
    :return: the loss and the EER for this batch of embeddings.
    """
    speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

    # sim_matrix
    sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                        speakers_per_batch))

    # target
    ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)

    inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
    labels = np.array([inv_argmax(i) for i in ground_truth])
    preds = sim_matrix

    # Snippet from https://yangcha.github.io/EER-ROC/
    fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer