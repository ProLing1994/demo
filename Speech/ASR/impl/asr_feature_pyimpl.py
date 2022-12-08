import copy
# import numba as nb              # 有 bug
import numpy as np
# from scipy.fftpack import fft
from numpy.fft import fft
# from pyfftw.interfaces.numpy_fft import fft 


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.)


def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 **(mel / 2595.0) - 1)


def vtlp_augmentation(freqpoints, samplerate):
    '''
    vtlp: http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=34DDD4B0CDCE76942A879204E8B7716C?doi=10.1.1.369.733&rep=rep1&type=pdf
    '''
    # init param
    f_hi = 4800

    # alpha: 0.9 ~ 1.1
    alpha = np.random.uniform(0, 1) * 0.2 + 0.9

    # vtlp，freqpoints 添加扰动
    freqpoints_temp = copy.deepcopy(freqpoints)
    f = freqpoints[freqpoints <= f_hi * min(alpha, 1) / alpha]
    freqpoints_temp[freqpoints <= f_hi * min(alpha, 1) / alpha] = f * alpha

    f = freqpoints[freqpoints > f_hi * min(alpha, 1) / alpha]
    freqpoints_temp[freqpoints > f_hi * min(alpha, 1) / alpha] = samplerate / 2 - ((samplerate / 2 - f_hi * min(alpha, 1)) /
                                                            (samplerate / 2 - f_hi * min(alpha, 1) / alpha)) * (samplerate / 2 - f)

    freqpoints_temp[freqpoints_temp > freqpoints[-1]] = freqpoints[-1]
    freqpoints = freqpoints_temp


# @nb.jit()           # 运用 numba 加速 for 循环，有 bug
def get_frequency_feature(signal, sample_rate, winlen, winstep):
    """Compute FFT features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param sample_rate: the sample_rate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    """
    time_window = winlen * 1000 
    time_step = winstep * 1000  
    window_size = int(sample_rate * time_window / 1000)

    win_x = np.linspace(0, window_size - 1, window_size, dtype=np.int64)
    win_y = 0.54 - 0.46 * np.cos(2 * np.pi * win_x / (window_size - 1)) 
    signal_np = np.array(signal)

    range_end = int((len(signal) / float(sample_rate) * 1000 - time_window) // time_step) 
    data_input = np.zeros((range_end, int(window_size / 2)), dtype=np.float)  
    for i in range(0, range_end):
        p_start = int(i * sample_rate * time_step / 1000)
        p_end = p_start + window_size
        data_line = signal_np[p_start:p_end]
        if(data_line.shape[0] != win_y.shape[0]):
            continue
        data_line = data_line * win_y  # 加窗
        data_line = np.abs(fft(data_line)) / window_size
        data_input[i] = data_line[0:int(window_size / 2)] 
    return data_input


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None, bool_vtlp_augmentation=False):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :param bool_vtlp_augmentation: if this is true, Vocal Tract Length Perturbation (VTLP) augmentation.
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    # our points are in Hz, but we use fft bins, so we have to convert from Hz to fft bin number
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    freqpoints = mel2hz(melpoints)

    if bool_vtlp_augmentation:
        vtlp_augmentation(freqpoints, samplerate)

    bin = np.floor((nfft + 1) * freqpoints / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1]-bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2]-bin[ j+ 1])
    return fbank


class Feature(object):
    """ feature python wrapper """
    def __init__(self, sample_rate=16000, feature_freq=48, nfilt=64, winlen=0.032, winstep=0.010, scale_num=10):
        self.sample_rate = sample_rate          # 采样率
        self.feature_freq = feature_freq        # 特征个数
        self.nfilt = nfilt                      # mel 滤波器个数
        self.winlen = winlen                    # 窗宽
        self.winstep = winstep                  # 窗移
        self.scale_num = scale_num              # 缩放

        self.lowfreq = 10                       # 低频阈值
        self.highfreq = None                    # 高频阈值
        self.highfreq = self.highfreq or self.sample_rate / 2

        # init
        self.init()

    def init(self):
        # nfft 
        if self.sample_rate == 32000:           # fft 个数，跟采样率相关
            self.nfft = 1024                    
        elif self.sample_rate == 16000:
            self.nfft = 512 
        elif self.sample_rate == 8000:  
            self.nfft = 256  
        else:
            raise Exception("[ERROR] Unknow sample rate: {}/[8000, 16000, 32000]".format(self.sample_rate))
        
        # fb，mel 滤波器，跟采样率相关
        self.fb = get_filterbanks(self.nfilt, self.nfft, self.sample_rate, self.lowfreq, self.highfreq, False)
        if(self.sample_rate == 8000):
            self.fb = self.fb[:, :128]
        elif(self.sample_rate == 16000):
            self.fb = self.fb[:, :256]
        elif(self.sample_rate == 32000):
            self.fb = self.fb[:, :512]
        else:
            raise Exception("[ERROR] Unknow sample rate: {}/[8000, 16000, 32000]".format(self.sample_rate))

    def fbank(self, data, bool_vtlp_augmentation=False):
        """Compute Mel-filterbank features from an audio signal.
        """
        self.sample_rate, self.winlen, self.winstep, self.nfilt, self.nfft, self.lowfreq, self.highfreq,

        if bool_vtlp_augmentation:
            self.fb = get_filterbanks(self.nfilt, self.nfft, self.sample_rate, self.lowfreq, self.highfreq, bool_vtlp_augmentation)
            if(self.sample_rate == 8000):
                self.fb = self.fb[:, :128]
            elif(self.sample_rate == 16000):
                self.fb = self.fb[:, :256]
            elif(self.sample_rate == 32000):
                self.fb = self.fb[:, :512]
            else:
                raise Exception("[ERROR] Unknow sample rate: {}/[8000, 16000, 32000]".format(self.sample_rate))

        pspec = get_frequency_feature(data, self.sample_rate, self.winlen, self.winstep)
        feat = np.dot(pspec, self.fb.T) # compute the filterbank energies
        feat = np.where(feat == 0, np.finfo(float).eps, feat) # if feat is zero, we get problems with log
        return feat

    def __del__(self):
        pass
  
    def data_mat_time(self):
        pass
        
    def feature_time(self):
        pass

    def feature_freq(self):
        pass

    def get_mel_feature(self, data, bool_vtlp_augmentation=False):
        self.mel_feature = self.fbank(data, bool_vtlp_augmentation)
        self.mel_feature = self.mel_feature[:, :self.feature_freq]
        self.mel_feature = np.log(1 + self.mel_feature)
        return

    def get_mel_int_feature(self, data, bool_vtlp_augmentation=False):
        self.get_mel_feature(data, bool_vtlp_augmentation)

        self.mel_int_feature = self.mel_feature * 255 / self.scale_num  
        self.mel_int_feature = self.mel_int_feature.astype(int)
        temp_where = np.where(self.mel_int_feature > 255)
        self.mel_int_feature[temp_where] = 255
        temp_where = np.where(self.mel_int_feature < 0)
        self.mel_int_feature[temp_where] = 0
        self.mel_int_feature = self.mel_int_feature.astype(np.uint8)
        return

    def copy_mfsc_feature_to(self):
        return self.mel_feature

    def copy_mfsc_feature_int_to(self):
        return self.mel_int_feature