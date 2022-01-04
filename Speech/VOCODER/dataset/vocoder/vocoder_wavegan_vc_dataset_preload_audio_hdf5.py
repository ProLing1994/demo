from multiprocessing import Manager
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
from Basic.utils.hdf5_tools import *

from VC.dataset.cyclevae.dataset_preload_audio_hdf5 import load_data_pd


class VocoderWaveGanVcDataset(Dataset):
    def __init__(self, cfg, mode, augmentation_on=True, bool_return_name=False):
        # init
        self.cfg = cfg
        self.mode = mode
        self.augmentation_on = augmentation_on
        self.bool_return_name = bool_return_name
        self.hop_size = self.cfg.dataset.hop_size
        self.allow_cache = self.cfg.dataset.allow_cache

        self.data_pd = load_data_pd(cfg, mode)

        # filter by threshold
        self.filter_by_threshold()
        # assert the number of files
        assert len(self.data_pd) != 0, f"Not found any audio files."

        if self.cfg.dataset.normalize_bool:
            # restore scaler
            self.scaler = StandardScaler()

            dataset_name = '_'.join(cfg.general.dataset_list)
            self.stats_jnt_path = os.path.join(cfg.general.data_dir, 'dataset_audio_normalize_hdf5', dataset_name, 'world', f"stats_jnt.h5")
            self.scaler.mean_ = read_hdf5(self.stats_jnt_path, "mean_feat_org_lf0")
            self.scaler.scale_ = read_hdf5(self.stats_jnt_path, "scale_feat_org_lf0")

            # from version 0.23.0, this information is needed
            self.scaler.n_features_in_ = self.scaler.mean_.shape[0]

        if self.allow_cache:
            # NOTE: Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.data_pd))]

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, index):

        if self.allow_cache and len(self.caches[index]) != 0:
            return self.caches[index][0], self.caches[index][1]
        
        dataset_name = self.data_pd.loc[index, 'dataset']
        key_name = self.data_pd.loc[index, 'key']

        # state
        state_path = self.data_pd.loc[index, 'state']

        # wav
        wav = read_hdf5(state_path, "wave")

        # mel (T, C)
        mel = read_hdf5(state_path, "feat_org_lf0")

        # normalize
        if self.cfg.dataset.normalize_bool:
            mel = self.scaler.transform(mel)

        # Quantize the wav         
        # 注意：hdf5 格式，采用 fbank_nopreemphasis_log_manual 计算方式，不进行预加重处理
        # wav = audio.compute_pre_emphasis(wav)
        # wav = np.clip(wav, -1, 1)

        # Fix for missing padding   # TODO: settle on whether this is any useful
        r_pad =  (len(wav) // self.hop_size + 1) * self.hop_size - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        wav = wav[: len(mel) * self.hop_size]
        # make sure the audio length and feature length are matched
        assert len(mel) * self.hop_size == len(wav)

        if self.allow_cache:
            self.caches[index] = (wav, mel)

        if not self.bool_return_name:
            return wav, mel
        else:
            return wav, mel, key_name

    def filter_by_threshold(self):
        self.drop_index_list = []
        audio_length_threshold = int(self.cfg.dataset.sampling_rate * self.cfg.dataset.clip_duration_ms / 1000)

        for index, row in self.data_pd.iterrows():

            # state
            state_path = self.data_pd.loc[index, 'state']

            # wav
            wav = read_hdf5(state_path, "wave")
            wav_length = len(wav)

            if wav_length < audio_length_threshold:
                self.drop_index_list.append(index)

        if len(self.drop_index_list) != 0:
            print(f"[Warning] Some files are filtered by audio length threshold ({len(self.data_pd)} -> {len(self.data_pd) - len(self.drop_index_list)}).")
        
        # drop
        self.data_pd.drop(self.drop_index_list, inplace=True)
        self.data_pd.reset_index(drop=True, inplace=True)


class VocoderVcCollater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(self, cfg):
        """Initialize customized collater for PyTorch DataLoader."""

        self.hop_size = cfg.dataset.hop_size
        self.batch_max_steps = int(cfg.dataset.sampling_rate * cfg.dataset.clip_duration_ms / 1000)

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

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Target signal batch (B, 1, T).

        """
        
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
        y_batch = np.array([x[start:end] for x, start, end in zip(xs, x_starts, x_ends)])
        c_batch = np.array([c[start:end] for c, start, end in zip(cs, c_starts, c_ends)])

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
        # filter by threshold
        # 确保 mel 频率的长度，保存可以正常切片
        assert len(c) > self.mel_threshold

        if len(x) < len(c) * self.hop_size:
            x = np.pad(x, (0, len(c) * self.hop_size - len(x)), mode="edge")

        # check the legnth is valid
        assert len(x) == len(c) * self.hop_size

        return x, c 