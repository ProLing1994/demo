import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt

# 绘制时域图
def plot_time(signal, sample_rate, title='Audio Image'):
  time = np.arange(0, len(signal)) * (1.0 / sample_rate)
  plt.figure(figsize=(10, 4))
  plt.plot(time, signal)
  plt.xlabel('Time(s)')
  plt.ylabel('Amplitude')
  plt.title(title)
  plt.grid()
  plt.show()

if __name__ == "__main__":
  audio_path = "/home/huanyuan/data/speech/canting.wav"
  
  # options
  pre_emphasis = 0.97
  frame_size, frame_stride = 0.025, 0.01

  # 1. 读取音频数据
  sample_rate, signal = wavfile.read(audio_path)
  print('sample rate:', sample_rate, ', frame length:', len(signal))
  assert sample_rate == 16000, "[ERROR: Sample Rate {} != 16000]".format(sample_rate)
  
  # 绘制原始音频时域图
  plot_time(signal, sample_rate, 'Original Audio Image') 

  # 2. 预加重(Pre-Emphasis)
  emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

  # 绘制预加重音频时域图
  # plot_time(emphasized_signal, sample_rate, 'Pre-Emphasis Audio Image')

  # 3. 分帧
  frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
  signal_length = len(emphasized_signal)
  num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1    # 帧数
  num_frames_per_second = int(np.ceil(np.abs(sample_rate - frame_length) / frame_step)) + 1

  pad_signal_length = (num_frames - 1) * frame_step + frame_length
  z = np.zeros((pad_signal_length - signal_length))
  pad_signal = np.append(emphasized_signal, z)

  indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1, 1)
  frames = pad_signal[indices]
  print('frames shape:', frames.shape)

  # 4. 加窗(汉宁窗)
  hamming = np.hamming(frame_length)
  # hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, frame_length) / (frame_length - 1))

  # plt.figure(figsize=(20, 5))
  # plt.plot(hamming)
  # plt.grid()
  # plt.xlim(0, 200)
  # plt.ylim(0, 1)
  # plt.xlabel('Samples')
  # plt.ylabel('Amplitude')
  # plt.show()

  frames *= hamming
  # 绘制加窗后第一帧音频时域图
  # plot_time(frames[0], sample_rate, 'Frame 0: Hamming Audio Image')

  # 5. 短时能量谱(Short term energy)
  # 绘制第一帧音频能量图
  plot_time(np.square(frames[0]), sample_rate, 'Frame 0: Short Term Energy Image')

  frames_short_term_energy = np.square(frames)
  frames_short_term_energy = np.sum(frames_short_term_energy, axis = 1)  
  frames_short_term_energy = frames_short_term_energy * 1.0 / max(frames_short_term_energy)
  print('frames_short_term_energy shape:', frames_short_term_energy.shape)
  plot_time(frames_short_term_energy, num_frames_per_second, 'Short Term Energy Image')