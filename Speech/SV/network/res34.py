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

    
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                                padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes, affine=False)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=1, \
                                padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, affine=False)

        self.relu = torch.nn.ReLU()
        self.shortcut = nn.Sequential()
        
        # 经过处理后的 x 要与 x 的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积 +BN 来变换为同一维度
        if stride != [1, 1] or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes, affine=False)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SpeakerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # init
        model_embedding_size = 256
        filters_list = [32, 64, 128, 256]
        strides_list = [1, 2, 2, 2]
        residual_units_num_list = [3, 4, 6, 3]
        
        self.speakers_per_batch = cfg.train.speakers_per_batch if 'speakers_per_batch' in cfg.train else None
        self.utterances_per_speaker = cfg.train.utterances_per_speaker if 'utterances_per_speaker' in cfg.train else None
        self.model_embedding_size = model_embedding_size

        # Network defition
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        layers = []
        layers.append(BasicBlock(16, filters_list[0], 3, strides_list[0], 1, 1))
        for _ in range(1, residual_units_num_list[0]):
            layers.append(BasicBlock(filters_list[0], filters_list[0], 3, 1, 1, 1))
        self.conv2 = nn.Sequential(*layers)

        layers = []
        layers.append(BasicBlock(filters_list[0], filters_list[1], 3, strides_list[1], 1, 1))
        for _ in range(1, residual_units_num_list[1]):
            layers.append(BasicBlock(filters_list[1], filters_list[1], 3, 1, 1, 1))
        self.conv3 = nn.Sequential(*layers)

        layers = []
        layers.append(BasicBlock(filters_list[1], filters_list[2], 3, strides_list[2], 1, 1))
        for _ in range(1, residual_units_num_list[2]):
            layers.append(BasicBlock(filters_list[2], filters_list[2], 3, 1, 1, 1))
        self.conv4 = nn.Sequential(*layers)

        layers = []
        layers.append(BasicBlock(filters_list[2], filters_list[3], 3, strides_list[3], 1, 1))
        for _ in range(1, residual_units_num_list[3]):
            layers.append(BasicBlock(filters_list[3], filters_list[3], 3, 1, 1, 1))
        self.conv5 = nn.Sequential(*layers)

        self.linear1 = nn.Linear(in_features=filters_list[3], 
                                out_features=256)
        self.linear2 = nn.Linear(in_features=256, 
                                out_features=self.model_embedding_size)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        utterances = utterances.reshape(utterances.shape[0], 1, utterances.shape[1], utterances.shape[2])     # shape: (batch, 401, 80) ->  shape: (batch, 1, 401, 80)

        out = self.conv1(utterances)    # shape: (batch, 1, 401, 80)->  shape: (batch, 16, 401, 80)
        out = self.conv2(out)           # shape: (batch, 16, 401, 80)->  shape: (batch, 32, 401, 80)
        out = self.conv3(out)           # shape: (batch, 16, 401, 80)->  shape: (batch, 64, 201, 40)
        out = self.conv4(out)           # shape: (batch, 64, 201, 40)->  shape: (batch, 128, 101, 20)
        out = self.conv5(out)           # shape: (batch, 128, 101, 20)->  shape: (batch, 256, 51, 10)
        
        out = out.view(out.size(0), out.size(1), -1)  # shape: (batch, 256, 51, 10)->  shape: (batch, 256, 510)
        out = torch.mean(out, 2)        # shape: (batch, 256, 510)->  shape: (batch, 256)
        # We take only the hidden state of the last layer
        out = self.linear1(out)         # shape: (batch, 256, 510)->  shape: (batch, 256)
        out = self.dropout(out)         # shape: (batch, 256)->  shape: (batch, 256)
        
        embeds_raw = self.relu(self.linear2(out))       # shape: (batch, 256)->  shape: (batch, 256)

        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        
        return embeds
    
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