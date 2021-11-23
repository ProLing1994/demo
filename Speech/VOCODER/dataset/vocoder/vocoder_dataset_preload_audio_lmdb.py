import os
import pandas as pd
import sys
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.dataset import audio
from Basic.utils.lmdb_tools import *

from TTS.dataset.text.text import *
from TTS.dataset.tts.audio import *
from TTS.dataset.tts.sv2tts_dataset_preload_audio_lmdb import *

from VOCODER.dataset.vocoder.audio import *
import VOCODER.config.vocoder.hparams as hparams_vocoder


class VocoderDataset(Dataset):
    def __init__(self, cfg, mode, augmentation_on=True):
        # init
        self.cfg = cfg
        self.mode = mode
        self.augmentation_on = augmentation_on

        self.data_pd = load_data_pd(cfg, mode)
        self.data_list = self.data_pd['unique_utterance'].to_list()
        if len(self.data_list) == 0:
            raise Exception("No speakers found. ")
        
        self.hop_length = int(self.cfg.dataset.sample_rate * self.cfg.dataset.window_stride_ms / 1000)
        # print("Found %d samples. " % len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if not hasattr(self, 'lmdb_dict'):
            self.lmdb_dict = load_lmdb(self.cfg, self.mode)

        lmdb_dataset = self.data_pd.loc[index, 'dataset']
        data_name = self.data_pd.loc[index, 'unique_utterance']

        if self.cfg.dataset.language == 'chinese':
            text = self.data_pd.loc[index, 'pinyin']
        elif self.cfg.dataset.language == 'english':
            text = self.data_pd.loc[index, 'text']
        else:
            raise Exception("[ERROR:] Unknow dataset language: {}".format(self.cfg.dataset.language))

        # text
        # Get the text and clean it
        text = text_to_sequence(text, self.cfg.dataset.tts_cleaner_names, lang=self.cfg.dataset.symbols_lang)
        
        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)

        # mel
        wav = read_audio_lmdb(self.lmdb_dict[lmdb_dataset], str(data_name))
        mel = audio.compute_mel_spectrogram(self.cfg, wav).T.astype(np.float32)

        # embed wav
        embed_wav = wav.copy()

        # Quantize the wav
        wav = audio.compute_pre_emphasis(wav)
        wav = np.clip(wav, -1, 1)

        # Fix for missing padding   # TODO: settle on whether this is any useful
        r_pad =  (len(wav) // self.hop_length + 1) * self.hop_length - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')

        if hparams_vocoder.voc_mode == 'RAW':
            if hparams_vocoder.mu_law:
                quant = encode_mu_law(wav, mu=2 ** hparams_vocoder.voc_bits)
            else:
                quant = float_2_label(wav, bits=hparams_vocoder.voc_bits)
        elif hparams_vocoder.voc_mode == 'MOL':
            quant = float_2_label(wav, bits=16)
        return text, mel, embed_wav, quant.astype(np.int64)


class VocoderDataLoader(DataLoader):
    def __init__(self, dataset, sampler, cfg):
        super().__init__(
            dataset=dataset, 
            batch_size=cfg.train.batch_size, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=None, 
            num_workers=cfg.train.num_threads,
            collate_fn=lambda data: self.collate_vocoder(data, cfg),
            pin_memory=True, 
            drop_last=False, 
            timeout=0, 
            worker_init_fn=None
        )

    def collate_vocoder(self, data, cfg):
        # Text
        char_lengths = [len(x[0]) for x in data]
        max_char_length = max(char_lengths)

        chars = [pad1d(x[0], max_char_length) for x in data]
        chars = np.stack(chars)

        # Mel spectrogram
        mel_lengths = [x[1].shape[1] for x in data]
        max_spec_len = max(mel_lengths)
        if max_spec_len % cfg.net.r != 0:
            max_spec_len += cfg.net.r - max_spec_len % cfg.net.r

        # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
        # By default, TTS uses symmetric mels, where -1*max_abs_value is silence.
        if hparams.symmetric_mels:
            mel_pad_value = -1 * hparams.max_abs_value
        else:
            mel_pad_value = 0

        mels = [pad2d(x[1], max_spec_len, pad_value=mel_pad_value) for x in data]
        mels = np.stack(mels)

        # Speaker embedding (SV2TTS)
        embed_wavs = [x[2] for x in data]

        # quant (vocoder)
        quants = [x[3] for x in data]

        # Convert all to tensor
        chars = torch.tensor(chars).long()
        char_lengths = torch.tensor(char_lengths)
        mels = torch.tensor(mels)
        mel_lengths = torch.tensor(mel_lengths)
        return chars, char_lengths, mels, mel_lengths, embed_wavs, quants


def prepare_data(cfg, mel_out, mel, quant):
    # init 
    hop_length = int(cfg.dataset.sample_rate * cfg.dataset.window_stride_ms / 1000)

    mel_list = []
    quant_list = []

    for idx in range(len(quant)):
        # mel
        mel_idx = mel_out[idx].T[:int(mel[idx])]
        mel_idx = mel_idx.T.astype(np.float32) / hparams.max_abs_value
        mel_list.append(mel_idx)
        
        # quant
        quant_idx = quant[idx]
        assert len(quant_idx) >= mel_idx.shape[1] * hop_length
        quant_idx = quant_idx[:mel_idx.shape[1] * hop_length]
        assert len(quant_idx) % hop_length == 0
        quant_list.append(quant_idx)

    voc_seq_len = hparams_vocoder.voc_seq_multiple * hop_length
    mel_win = voc_seq_len // hop_length + 2 * hparams_vocoder.voc_pad
    max_offsets = [x.shape[-1] -2 - (mel_win + 2 * hparams_vocoder.voc_pad) for x in mel_list]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hparams_vocoder.voc_pad) * hop_length for offset in mel_offsets]

    mels = [x[:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(mel_list)]
    labels = [x[sig_offsets[i]:sig_offsets[i] + voc_seq_len + 1] for i, x in enumerate(quant_list)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, : voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hparams_vocoder.voc_mode == 'MOL' else hparams_vocoder.voc_bits
    
    x = label_2_float(x.float(), bits)
    if hparams_vocoder.voc_mode == 'MOL' :
        y = label_2_float(y.float(), bits)

    return x, y, mels, mel_list, quant_list