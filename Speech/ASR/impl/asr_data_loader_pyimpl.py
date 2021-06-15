import librosa
import numpy as np
import os
import soundfile as sf

class WaveLoader_Soundfile(object):
    """ wav loader python wrapper """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __del__(self):
        pass

    def load_data(self, audio_path="/home/huanyuan/share/audio_data/xiaorui_12162020_training_60_001.wav"):
        self.data, _ = sf.read(audio_path, dtype='int16')
        return 

    def save_data(self, data, output_path):
        sf.write(output_path, data, self.sample_rate)
        return

    def data_length(self):
        return self.data.shape[0] / self.sample_rate

    def to_numpy(self):
        return self.data

class WaveLoader_Librosa(object):
    """ wav loader python wrapper """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __del__(self):
        pass

    def load_data(self, audio_path="/home/huanyuan/share/audio_data/xiaorui_12162020_training_60_001.wav"):
        self.data = librosa.core.load(audio_path, sr=self.sample_rate)[0]
        return 

    def save_data(self, data, output_path):
        data = data.astype(np.float32)
        audio_sample = data / float(pow(2, 15))
        temp_path = os.path.join(os.path.dirname(output_path), '{}.wav'.format('temp'))
        librosa.output.write_wav(temp_path, audio_sample, sr=self.sample_rate)
        os.system('sox {} -b 16 -e signed-integer {}'.format(temp_path, output_path))
        return

    def data_length(self):
        return self.data.shape[0] / self.sample_rate

    def to_numpy(self):
        data = self.data * pow(2,15)
        data = data.astype(int)
        return data
