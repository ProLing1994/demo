import sys
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')

from TTS.dataset.tts.sv2tts_dataset_preload_audio_lmdb import *


class VocoderWaveGanDataset(Dataset):
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
        
        # wav
        wav = read_audio_lmdb(self.lmdb_dict[lmdb_dataset], str(data_name))

        # mel
        mel = audio.compute_mel_spectrogram(self.cfg, wav).T.astype(np.float32).T

        # Quantize the wav          # TODO：不进行预加重，注：解码时也要进行反预加重
        wav = audio.compute_pre_emphasis(wav)
        wav = np.clip(wav, -1, 1)

        # # Quantize the mel          # 不进行缩放，注：解码时也不进行缩放
        # mel = mel.astype(np.float32) / hparams.max_abs_value

        # Fix for missing padding   # TODO: settle on whether this is any useful
        r_pad =  (len(wav) // self.hop_length + 1) * self.hop_length - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        wav = wav[: len(mel) * self.hop_length]
        # make sure the audio length and feature length are matched
        assert len(mel) * self.hop_length == len(wav)

        return wav, mel


class VocoderWaveGanDataLoader(DataLoader):
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

    def collate_vocoder(self, batch, cfg):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Target signal batch (B, 1, T).

        """
        # init 
        if not hasattr(self, 'hop_size'):
            self.hop_size = int(cfg.dataset.sample_rate * cfg.dataset.window_stride_ms / 1000)
            self.batch_max_steps = int(cfg.dataset.sample_rate * cfg.dataset.clip_duration_ms / 1000)

            if self.batch_max_steps % self.hop_size != 0:
                self.batch_max_steps += -(self.batch_max_steps % self.hop_size)
            assert self.batch_max_steps % self.hop_size == 0
            self.batch_max_frames = self.batch_max_steps // self.hop_size

            self.aux_context_window = cfg.net.yaml["generator_params"].get("aux_context_window", 0)
            self.use_noise_input = cfg.net.yaml["use_noise_input"]

            # set useful values in random cutting
            self.start_offset = self.aux_context_window
            self.end_offset = -(self.batch_max_frames + self.aux_context_window)
            self.mel_threshold = self.batch_max_frames + 2 * self.aux_context_window

        # check length
        batch = [self._adjust_length(*b) for b in batch]
        xs, cs = [b[0] for b in batch], [b[1] for b in batch]

        # make batch with random cut
        c_lengths = [len(c) for c in cs]
        start_frames = np.array(
            [
                np.random.randint(self.start_offset, cl + self.end_offset)
                for cl in c_lengths
            ]
        )
        x_starts = start_frames * self.hop_size
        x_ends = x_starts + self.batch_max_steps
        c_starts = start_frames - self.aux_context_window
        c_ends = start_frames + self.batch_max_frames + self.aux_context_window
        y_batch = [x[start:end] for x, start, end in zip(xs, x_starts, x_ends)]
        c_batch = [c[start:end] for c, start, end in zip(cs, c_starts, c_ends)]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)
        c_batch = torch.tensor(c_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')

        # make input noise signal batch tensor
        if self.use_noise_input:
            z_batch = torch.randn(y_batch.size())  # (B, 1, T)
            return (z_batch, c_batch), y_batch
        else:
            return (c_batch,), y_batch

    def _adjust_length(self, x, c):
        """Adjust the audio and feature lengths.

        Note:
            Basically we assume that the length of x and c are adjusted
            through preprocessing stage, but if we use other library processed
            features, this process will be needed.

        """
        assert len(c) > self.mel_threshold

        if len(x) < len(c) * self.hop_size:
            x = np.pad(x, (0, len(c) * self.hop_size - len(x)), mode="edge")

        # check the legnth is valid
        assert len(x) == len(c) * self.hop_size

        return x, c