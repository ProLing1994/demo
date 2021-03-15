import librosa

class WaveLoader(object):
    """ wav loader python wrapper """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __del__(self):
        pass

    def load_data(self, audio_path="/home/huanyuan/share/audio_data/xiaorui_12162020_training_60_001.wav"):
        self.data = librosa.core.load(audio_path, sr=self.sample_rate)[0]
        return 

    def data_length(self):
        pass

    def to_numpy(self):
        return self.data * pow(2,15)
