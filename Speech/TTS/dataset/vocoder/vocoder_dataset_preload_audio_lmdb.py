import os
import pandas as pd
import sys
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.utils.lmdb_tools import *

from TTS.dataset.text import *
from TTS.dataset.audio import *
from TTS.dataset.vocoder.audio import *
from TTS.dataset.sv2tts_dataset_preload_audio_lmdb import *

class VocoderDataset(Dataset):
    def __init__(self, cfg, mode, augmentation_on=True):
        # init
        self.cfg = cfg
        self.mode = mode
        self.augmentation_on = augmentation_on

        self.hop_length = int(self.cfg.dataset.sample_rate * self.cfg.dataset.window_stride_ms / 1000)
        self.data_pd = load_data_pd(cfg, mode)
        self.data_list = self.data_pd['sub_basename'].to_list()
        if len(self.data_list) == 0:
            raise Exception("No speakers found. ")
        
        print("Found %d samples. " % len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if not hasattr(self, 'lmdb_dict'):
            self.lmdb_dict = load_lmdb(self.cfg, self.mode)

        lmdb_dataset = self.data_pd.loc[index, 'dataset']
        data_name = self.data_pd.loc[index, 'sub_basename']
        text = self.data_pd.loc[index, 'text']

        # text
        # Get the text and clean it
        text = text_to_sequence(text, self.cfg.dataset.tts_cleaner_names)
        
        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)

        # mel
        wav = read_audio_lmdb(self.lmdb_dict[lmdb_dataset], str(data_name))
        mel = audio_preprocess(self.cfg, wav).T.astype(np.float32)

        # stop length
        mel_frames = mel.shape[1]

        # embed wav
        embed_wav = preprocess_wav(wav, self.cfg.dataset.sample_rate)

        # Quantize the wav
        if preemphasize:
            wav = pre_emphasis(wav)
        wav = np.clip(wav, -1, 1)

        # Fix for missing padding   # TODO: settle on whether this is any useful
        r_pad =  (len(wav) // self.hop_length + 1) * self.hop_length - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')

        if voc_mode == 'RAW':
            if mu_law:
                quant = encode_mu_law(wav, mu=2 ** bits)
            else:
                quant = float_2_label(wav, bits=bits)
        elif voc_mode == 'MOL':
            quant = float_2_label(wav, bits=16)
        return text, mel, mel_frames, embed_wav, quant.astype(np.int64)


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
        x_lens = [len(x[0]) for x in data]
        max_x_len = max(x_lens)

        chars = [pad1d(x[0], max_x_len) for x in data]
        chars = np.stack(chars)

        # Mel spectrogram
        spec_lens = [x[1].shape[-1] for x in data]
        max_spec_len = max(spec_lens) + 1 
        if max_spec_len % cfg.net.r != 0:
            max_spec_len += cfg.net.r - max_spec_len % cfg.net.r

        # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
        # By default, SV2TTS uses symmetric mels, where -1*max_abs_value is silence.
        if symmetric_mels:
            mel_pad_value = -1 * max_abs_value
        else:
            mel_pad_value = 0

        mel = [pad2d(x[1], max_spec_len, pad_value=mel_pad_value) for x in data]
        mel = np.stack(mel)
        
        # Mel Frames: stop
        mel_frames = [x[2] for x in data]

        # Speaker embedding (SV2TTS)
        embed_wav = [x[3] for x in data]

        # quant (vocoder)
        quant = [x[4] for x in data]

        # Convert all to tensor
        chars = torch.tensor(chars).long()
        mel = torch.tensor(mel)
        return chars, mel, mel_frames, embed_wav, quant