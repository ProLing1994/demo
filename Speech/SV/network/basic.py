import numpy as np
import sys
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, '/home/huanyuan/code/demo/common')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/common')
from common.utils.python.kaiming_init import kaiming_weight_init


def parameters_init(net):
      net.apply(kaiming_weight_init)


class SpeakerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # init
        mel_n_channels = cfg.dataset.feature_bin_count
        model_hidden_size = 256
        model_embedding_size = 256
        model_num_layers = 3
        self.num_classes = cfg.loss.num_classes
        self.method = cfg.loss.method

        self.speakers_per_batch = cfg.train.speakers_per_batch if 'speakers_per_batch' in cfg.train else None
        self.utterances_per_speaker = cfg.train.utterances_per_speaker if 'utterances_per_speaker' in cfg.train else None
        self.model_embedding_size = model_embedding_size

        # Network defition
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size)
        self.relu = torch.nn.ReLU()

        if self.method == 'softmax':
            self.linear2 = nn.Linear(in_features=self.model_embedding_size, 
                                    out_features=self.num_classes)

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))
        
    def do_gradient_ops(self):
        # Gradient scale
        if self.method == 'ge2e':
            self.similarity_weight.grad *= 0.01
            self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        self.lstm.flatten_parameters()
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        
            
        if self.method == 'ge2e':
            return embeds

        elif self.method == 'softmax':
            out = self.linear2(embeds)
            return embeds, out
        
        else:
            raise Exception("[Unknow:] cfg.loss.method. ")
    
    def embeds_view(self, embeds):
        embeds = embeds.view(-1, self.utterances_per_speaker, self.model_embedding_size)
        return embeds

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).cuda()
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           )
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix

    def similarity_matrix_cpu(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = np.mean(embeds, axis=1, keepdims=True)
        centroids_incl = centroids_incl / (np.linalg.norm(centroids_incl, axis=2, keepdims=True) + 1e-5)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (np.sum(embeds, axis=1, keepdims=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl / (np.linalg.norm(centroids_excl, axis=2, keepdims=True) + 1e-5)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = np.zeros((speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch))
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(axis=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(axis=1)
        
        sim_matrix = sim_matrix * self.similarity_weight.detach().cpu().numpy() + self.similarity_bias.detach().cpu().numpy()
        return sim_matrix