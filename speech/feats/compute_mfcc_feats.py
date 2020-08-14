import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt

# 绘制时域图
def plot_time(signal, sample_rate):
  time = np.arange(0, len(signal)) * (1.0 / sample_rate)
  plt.figure(figsize=(20, 5))
  plt.plot(time, signal)
  plt.xlabel('Time(s)')
  plt.ylabel('Amplitude')
  plt.grid()
  plt.show()

# 绘制频域图
def plot_freq(signal, sample_rate, fft_size=512):
  # 调用 np.fft 的函数 rfft(用于实值信号fft)，产生长度为 fft_size/2+1 的一个复数向量，分别表示从 0Hz~4000Hz 的部分
  # 这里之所以是 4000Hz 是因为 Nyquis t定理，采样频率 8000Hz，则能恢复带宽为 4000Hz 的信号。最后 /fft_size 是为了正确显示波形能量
  xf = np.fft.rfft(signal, fft_size) / fft_size
  freqs = np.linspace(0, int(sample_rate/2), int(fft_size/2 + 1)) 
  xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
  plt.figure(figsize=(20, 5))
  plt.plot(freqs, xfp)
  plt.xlabel('Freq(hz)')
  plt.ylabel('dB')
  plt.grid()
  plt.show()

# 绘制频谱图
def plot_spectrogram(spec, note):
  fig = plt.figure(figsize=(20, 5))
  heatmap = plt.pcolor(spec)
  fig.colorbar(mappable=heatmap)
  plt.xlabel('Time(s)')
  plt.ylabel(note)
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  wav_file = "/home/huanyuan/code/demo/speech/feats/OSR_us_000_0010_8k.wav"
  sample_rate, signal = wavfile.read(wav_file)
  signal = signal[0: int(3.5 * sample_rate)]    # 保留前3.5s数据
  print('sample rate:', sample_rate, ', frame length:', len(signal))
  
  # 绘制时域图
  plot_time(signal, sample_rate) 
  # 绘制频域图
  plot_freq(signal, sample_rate)
  
  # 预加重（Pre-Emphasis）
  pre_emphasis = 0.97
  emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

  plot_time(emphasized_signal, sample_rate)
  plot_freq(emphasized_signal, sample_rate)

  # 分帧
  frame_size, frame_stride = 0.025, 0.01
  frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
  signal_length = len(emphasized_signal)
  num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1    # 帧数

  pad_signal_length = (num_frames - 1) * frame_step + frame_length
  z = np.zeros((pad_signal_length - signal_length))
  pad_signal = np.append(emphasized_signal, z)

  indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1, 1)
  frames = pad_signal[indices]
  print(frames.shape)

  # 加窗
  hamming = np.hamming(frame_length)
  # hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, frame_length) / (frame_length - 1))

  plt.figure(figsize=(20, 5))
  plt.plot(hamming)
  plt.grid()
  plt.xlim(0, 200)
  plt.ylim(0, 1)
  plt.xlabel('Samples')
  plt.ylabel('Amplitude')
  plt.show()

  frames *= hamming
  plot_time(frames[1], sample_rate)
  plot_freq(frames[1], sample_rate)

  # 快速傅里叶变换（FFT）
  NFFT = 512
  mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
  pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
  print(pow_frames.shape)
  plt.figure(figsize=(20, 5))
  plt.plot(pow_frames[1])
  plt.grid()

  # Mel 滤波器组
  low_freq_mel = 0
  high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
  print(low_freq_mel, high_freq_mel)

  nfilt = 40
  mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
  hz_points = 700 * (10 ** (mel_points / 2595) - 1)

  fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
  bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
  for i in range(1, nfilt + 1):
      left = int(bin[i-1])
      center = int(bin[i])
      right = int(bin[i+1])
      for j in range(left, center):
          fbank[i-1, j+1] = (j + 1 - bin[i-1]) / (bin[i] - bin[i-1])
      for j in range(center, right):
          fbank[i-1, j+1] = (bin[i+1] - (j + 1)) / (bin[i+1] - bin[i])
  print(fbank)

  filter_banks = np.dot(pow_frames, fbank.T)
  filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
  filter_banks = 20 * np.log10(filter_banks)  # dB
  print(filter_banks.shape)
  plot_spectrogram(filter_banks.T, 'Filter Banks')

  # 离散余弦变换 dct 
  num_ceps = 12
  mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps+1)]
  print(mfcc.shape)
  plot_spectrogram(mfcc.T, 'MFCC Coefficients')
  print()