from math import sqrt
import json
import sys
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')

from TTS.network.sv2tts.tacotron2 import tacotron_config, Encoder, Decoder, Postnet
from TTS.network.sv2tts.modules import AdversarialClassifier


class Tacotron2(nn.Module):
    def __init__(self, cfg):

        super(Tacotron2, self).__init__()

        with open(tacotron_config, 'r') as f:
            model_cfg = json.load(f)

        self.cfg = cfg
        self.mel_dim = cfg.dataset.feature_bin_count 
        n_vocab = cfg.dataset.num_chars
        n_speaker = cfg.dataset.num_speakers
        self.r = cfg.net.r

        normal_cfg = model_cfg["normal"]
        max_decoder_steps = normal_cfg["max_decoder_steps"]
        stop_threshold = normal_cfg["stop_threshold"]

        # Embedding
        text_embedding_cfg = model_cfg["text_embedding"]
        text_embed_dim = text_embedding_cfg["text_embed_dim"]
        self.embedding = nn.Embedding(n_vocab, text_embed_dim)
        std = sqrt(2.0 / (n_vocab + text_embed_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        # Speaker Embedding
        spk_embedding_cfg = model_cfg["spk_embedding"]
        speaker_embed_dim = spk_embedding_cfg["spk_embed_dim"]
        self.speaker_embedding = nn.Embedding(n_speaker, speaker_embed_dim)

        # Encoder
        encoder_cfg = model_cfg["encoder"]
        encoder_out_dim = encoder_cfg["blstm_units"]
        self.encoder = Encoder(text_embed_dim, **encoder_cfg)

        # Adversarial speaker classifier
        spk_classifier_cfg = model_cfg["spk_classifier"]
        self.speaker_classifier = AdversarialClassifier(encoder_out_dim, n_speaker, **spk_classifier_cfg)

        # Decoder
        encoder_out_dim = encoder_out_dim + speaker_embed_dim
        decoder_cfg = model_cfg["decoder"]
        self.decoder = Decoder(self.mel_dim, self.r, encoder_out_dim, **decoder_cfg,
            max_decoder_steps=max_decoder_steps, stop_threshold=stop_threshold)

        # Postnet
        postnet_cfg = model_cfg["postnet"]
        self.postnet = Postnet(self.mel_dim, **postnet_cfg)

    def parse_data_batch(self, batch):
        """Parse data batch to form inputs and targets for model training/evaluating
        """
        texts, text_lengths, mels, mel_lengths, stops, speaker_ids = batch

        texts = texts.cuda()
        text_lengths = text_lengths.cuda()
        mels = mels.cuda()
        mel_lengths = mel_lengths.cuda()
        stops = stops.cuda()
        speaker_ids = speaker_ids.cuda()

        return (texts, text_lengths, mels, mel_lengths, speaker_ids), (mels, stops, speaker_ids)

    def do_gradient_ops(self):
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 1.0)

    def reduction_factor(self):
        return self.r

    # def forward(self, inputs):
    def forward(self, inputs, input_lengths, mels, mels_lengths, speaker_ids, speaker_embedding=None):
        del mels_lengths
        del speaker_embedding

        B = inputs.size(0)

        # (B, T)
        inputs = self.embedding(inputs)

        # (B, T, embed_dim)
        encoder_outputs = self.encoder(inputs)

        # (B, T, n_speaker)
        speaker_outputs = self.speaker_classifier(encoder_outputs)

        # (B) -> (B, T, speaker_embed_dim)
        speaker_embeddings = self.speaker_embedding(speaker_ids)[:, None]
        speaker_embeddings = speaker_embeddings.repeat(1, encoder_outputs.size(1), 1)

        # (B, T, encoder_out_dim + speaker_embed_dim)
        encoder_outputs = torch.cat((encoder_outputs, speaker_embeddings), dim=2)

        # (B, T, mel_dim)
        if not mels is None:
            mels = mels.permute(0, 2, 1).contiguous()
        mel_outputs, stop_tokens, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=input_lengths)

        # Postnet processing
        mel_post = self.postnet(mel_outputs)
        mel_post = mel_outputs + mel_post

        mel_outputs = mel_outputs.permute(0, 2, 1).contiguous()
        mel_post = mel_post.permute(0, 2, 1).contiguous()
        return mel_outputs, mel_post, stop_tokens, speaker_outputs, alignments

    def inference(self, inputs, speaker_ids, speaker_embedding=None):
        # Only text inputs
        return self.forward(inputs, None, None, None, speaker_ids, speaker_embedding)


class Tacotron2Loss(nn.Module):
    def __init__(self, cfg):
        super(Tacotron2Loss, self).__init__()
        del cfg
        
        with open(tacotron_config, 'r') as f:
            model_cfg = json.load(f)

        spk_loss_cfg = model_cfg["spk_loss"]
        self.speaker_loss_weight = spk_loss_cfg["spk_loss_weight"]

    def forward(self, predicts, targets):
        mel_target, stop_target, speaker_target = targets
        mel_target.requires_grad = False
        stop_target.requires_grad = False
        speaker_target.requires_grad = False

        mel_predict, mel_post_predict, stop_predict, speaker_predict = predicts

        mel_loss = nn.MSELoss()(mel_predict, mel_target)
        post_loss = nn.MSELoss()(mel_post_predict, mel_target)
        stop_loss = nn.BCELoss()(stop_predict, stop_target)

        # Compute speaker adversarial loss
        #
        # The speaker adversarial loss should be computed against each element of the encoder output.
        #
        # In Google's paper (https://arxiv.org/abs/1907.04448), it is mentioned that:
        # 'We impose this adversarial loss separately on EACH ELEMENT of the encoded text sequence,...'
        #
        speaker_target = speaker_target.unsqueeze(1).repeat(1, speaker_predict.size(1)) # (B) -> (B, T)
        speaker_predict = speaker_predict.transpose(1, 2) # (B, T, n_speaker) -> (B, n_speaker, T)
        speaker_loss = nn.CrossEntropyLoss()(speaker_predict, speaker_target)

        return (mel_loss + post_loss + stop_loss + speaker_loss * self.speaker_loss_weight), mel_loss, post_loss, stop_loss, speaker_loss * self.speaker_loss_weight