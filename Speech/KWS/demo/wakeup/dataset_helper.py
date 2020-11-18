import librosa
import numpy as np
import pcen


SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'


def load_label_index(positive_label):
    # data index
    label_index = {}
    for index, positive_word in enumerate(positive_label):
        label_index[positive_word] = index + 2
    label_index.update({SILENCE_LABEL: SILENCE_INDEX,
                        UNKNOWN_WORD_LABEL: UNKNOWN_WORD_INDEX})
    return label_index


class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_dct_filters=40, n_mels=40, f_max=4000, f_min=20, n_fft=480, hop_length=160):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.dct_filters = librosa.filters.dct(n_dct_filters, n_mels)
        self.pcen_transform = pcen.StreamingPCENTransform(
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, trainable=True)

    def compute_mfccs(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(self.dct_filters, x)
                for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").astype(np.float32)
        return data

    def compute_fbanks(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        data = data.T
        return data

    def compute_pcen(self, data):
        data = torch.from_numpy(np.expand_dims(data, axis=0))
        data = self.pcen_transform(data)
        self.pcen_transform.reset()
        data = data.detach().numpy()
        data = data.reshape(-1, 40)
        return data