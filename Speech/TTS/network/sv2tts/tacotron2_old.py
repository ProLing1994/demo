from math import sqrt
import numpy as np
import sys
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from TTS.network.sv2tts import hparams_tacotron2


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, cfg):
        super(Postnet, self).__init__()
        # init
        self.n_mel_channels = cfg.dataset.feature_bin_count               # n_mel_channels = 80
        self.n_frames_per_step = cfg.net.r                                # r = 1 

        self.postnet_embedding_dim = hparams_tacotron2.postnet_embedding_dim
        self.postnet_kernel_size = hparams_tacotron2.postnet_kernel_size
        self.postnet_n_convolutions = hparams_tacotron2.postnet_n_convolutions

        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.n_mel_channels, self.postnet_embedding_dim,
                         kernel_size=self.postnet_kernel_size, stride=1,
                         padding=int((self.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(self.postnet_embedding_dim))
        )

        for i in range(1, self.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(self.postnet_embedding_dim,
                             self.postnet_embedding_dim,
                             kernel_size=self.postnet_kernel_size, stride=1,
                             padding=int((self.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(self.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(self.postnet_embedding_dim, self.n_mel_channels,
                         kernel_size=self.postnet_kernel_size, stride=1,
                         padding=int((self.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(self.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # init 
        self.encoder_n_convolutions = hparams_tacotron2.encoder_n_convolutions
        self.encoder_embedding_dim = hparams_tacotron2.encoder_embedding_dim
        self.encoder_kernel_size = hparams_tacotron2.encoder_kernel_size

        convolutions = []
        for _ in range(self.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(self.encoder_embedding_dim,
                         self.encoder_embedding_dim,
                         kernel_size=self.encoder_kernel_size, stride=1,
                         padding=int((self.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(self.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(self.encoder_embedding_dim,
                            int(self.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, speaker_embeds, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        # add embedding
        if speaker_embeds is not None:
            outputs = self.add_speaker_embedding(outputs, speaker_embeds)
        return outputs

    def inference(self, x, speaker_embeds):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        if speaker_embeds is not None:
            outputs = self.add_speaker_embedding(outputs, speaker_embeds)
        return outputs

    def add_speaker_embedding(self, x, speaker_embeds):
        # SV2TTS
        # The input x is the encoder output and is a 3D tensor with size (batch_size, num_chars, tts_embed_dims)
        # When training, speaker_embeds is also a 2D tensor with size (batch_size, speaker_embedding_size)
        #     (for inference, speaker_embeds is a 1D tensor with size (speaker_embedding_size))
        # This concats the speaker embedding for each char in the encoder output

        # Save the dimensions as human-readable names
        batch_size = x.size()[0]
        num_chars = x.size()[1]

        if speaker_embeds.dim() == 1:
            idx = 0
        else:
            idx = 1

        # Start by making a copy of each speaker embedding to match the input text length
        # The output of this has size (batch_size, num_chars * tts_embed_dims)
        speaker_embedding_size = speaker_embeds.size()[idx]
        e = speaker_embeds.repeat_interleave(num_chars, dim=idx)

        # Reshape it and transpose
        e = e.reshape(batch_size, speaker_embedding_size, num_chars)
        e = e.transpose(1, 2)

        # Concatenate the tiled speaker embedding with the encoder output
        x = torch.cat((x, e), 2)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        # init
        self.n_mel_channels = cfg.dataset.feature_bin_count               # n_mel_channels = 80
        self.n_frames_per_step = cfg.net.r                                # r = 1 

        self.speaker_embedding_size = cfg.net.speaker_embedding_size
        self.encoder_embedding_dim = hparams_tacotron2.encoder_embedding_dim
        self.attention_rnn_dim = hparams_tacotron2.attention_rnn_dim
        self.decoder_rnn_dim = hparams_tacotron2.decoder_rnn_dim
        self.prenet_dim = hparams_tacotron2.prenet_dim
        self.max_decoder_steps = hparams_tacotron2.max_decoder_steps
        self.gate_threshold = hparams_tacotron2.gate_threshold
        self.p_attention_dropout = hparams_tacotron2.p_attention_dropout
        self.p_decoder_dropout = hparams_tacotron2.p_decoder_dropout
        self.attention_dim = hparams_tacotron2.attention_dim
        self.attention_location_n_filters = hparams_tacotron2.attention_location_n_filters
        self.attention_location_kernel_size = hparams_tacotron2.attention_location_kernel_size

        self.prenet = Prenet(
            self.n_mel_channels * self.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            self.prenet_dim + self.encoder_embedding_dim + self.speaker_embedding_size,
            self.attention_rnn_dim)

        # 为 self.speaker_embedding_size 添加额外的线性层
        if self.speaker_embedding_size:
            self.attention_layer_linear_projection = LinearNorm(
                self.encoder_embedding_dim + self.speaker_embedding_size,
                self.encoder_embedding_dim) 

        self.attention_layer = Attention(
            self.attention_rnn_dim, self.encoder_embedding_dim,
            self.attention_dim, self.attention_location_n_filters,
            self.attention_location_kernel_size)

        # 为 self.speaker_embedding_size 添加额外的线性层
        if self.speaker_embedding_size:
            self.decoder_rnn_linear_projection = LinearNorm(
                self.attention_rnn_dim + self.encoder_embedding_dim + self.speaker_embedding_size,
                self.attention_rnn_dim + self.encoder_embedding_dim) 

        self.decoder_rnn = nn.LSTMCell(
            self.attention_rnn_dim + self.encoder_embedding_dim,
            self.decoder_rnn_dim, 1)

        # 为 self.speaker_embedding_size 添加额外的线性层
        if self.speaker_embedding_size:
            self.projection_linear_projection = LinearNorm(
                self.decoder_rnn_dim + self.encoder_embedding_dim + self.speaker_embedding_size,
                self.decoder_rnn_dim + self.encoder_embedding_dim) 

        self.linear_projection = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            self.n_mel_channels * self.n_frames_per_step)

        self.gate_layer = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)                                   # text_t

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim + self.speaker_embedding_size).zero_())

        self.memory = memory
        if self.speaker_embedding_size:
            self.processed_memory = self.attention_layer_linear_projection(memory)
            self.processed_memory = self.attention_layer.memory_layer(self.processed_memory)
        else:
            self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.reshape(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)                    # shape: [2, 388, 146] [b, mel_t, text_t]
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)                # shape: [2, 388]      [b, mel_t]
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()     # shape: [2, 388, 80]  [b, mel_t, mel_f]
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)                               # shape: [2, 80, 388]  [b, mel_f, mel_t]

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)
        self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        if self.speaker_embedding_size:
            decoder_input = self.decoder_rnn_linear_projection(decoder_input)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        if self.speaker_embedding_size:
            decoder_hidden_attention_context = self.projection_linear_projection(decoder_hidden_attention_context)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        gate_prediction = torch.sigmoid(gate_prediction)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        # 准备每帧预测的起止帧（mel 帧）
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)          # shape: [386, 2, 80]  [mel_t, b, mel_f]
        decoder_inputs = self.prenet(decoder_inputs)                                # shape: [386, 2, 256] [mel_t, b, f]

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]                        # shape: [2, 256] [b, f]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs.extend([gate_output.squeeze(1)] * self.n_frames_per_step)
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, cfg):
        super(Tacotron2, self).__init__()
        # init
        self.n_mel_channels = cfg.dataset.feature_bin_count               # n_mel_channels = 80
        self.n_frames_per_step = cfg.net.r                                # r = 1 
        self.n_symbols = cfg.dataset.num_chars

        self.mask_padding = hparams_tacotron2.mask_padding
        self.fp16_run = hparams_tacotron2.fp16_run
        self.symbols_embedding_dim = hparams_tacotron2.symbols_embedding_dim

        self.embedding = nn.Embedding(self.n_symbols, self.symbols_embedding_dim)
        std = sqrt(2.0 / (self.n_symbols + self.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder()
        self.decoder = Decoder(cfg)
        self.postnet = Postnet(cfg)

        self.num_params()

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
            # By default, SV2TTS uses symmetric mels, where -1*max_abs_value is silence.
            if hparams.symmetric_mels:
                mel_pad_value = -1.0 * hparams.max_abs_value
            else:
                mel_pad_value = 0.0

            outputs[0].data.masked_fill_(mask, mel_pad_value)
            outputs[1].data.masked_fill_(mask, mel_pad_value)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1.0)  # gate energies

        return outputs

    def do_gradient_ops(self):
        # Gradient clipping
        clip_grad_norm_(self.parameters(), hparams_tacotron2.grad_clip_thresh)

    def forward(self, texts, text_lengths, mels, mel_lengths, speaker_embeds=None):
        text_lengths, mel_lengths = text_lengths.data, mel_lengths.data

        embedded_inputs = self.embedding(texts).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, speaker_embeds, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            mel_lengths)

    def inference(self, texts, speaker_embeds=None):
        embedded_inputs = self.embedding(texts).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs, speaker_embeds)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Tacontron2 Parameters: %.3fM" % parameters)
        return parameters