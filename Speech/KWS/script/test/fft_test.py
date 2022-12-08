from scipy.fftpack import fft
import librosa
import numpy as np

def GetFrequencyFeature3(wavsignal, fs):
    time_window = 32
    window_size = int(fs*32/1000)
    time_step = 10

    x = np.linspace(0, window_size - 1, window_size, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * x / (window_size - 1))
    # window_length = fs / 1000 * time_window
    wav_arr = np.array(wavsignal)

    wav_length = wav_arr.shape[0]
    ww=len(wavsignal)
    range0_end = int((len(wavsignal) / float(fs) * 1000 - time_window) // time_step)
    #print(range0_end,len(wavsignal),fs,time_window,time_step)
    data_input = np.zeros((range0_end, int(window_size / 2)), dtype=np.float)
    for i in range(0, range0_end-1):
        p_start = int(i * fs*time_step/1000)
        p_end = p_start + window_size
        data_line = wav_arr[p_start:p_end]
        if(data_line.shape[0]!=w.shape[0]):
            continue
        data_line = data_line * w  # 鍔犵獥

        data_line = np.abs(fft(data_line)) / window_size

        data_input[i] = data_line[0:int(window_size / 2)]
    #data_input = np.log(data_input + 1)

    return data_input

if __name__ == "__main__":
    input_wav = "/mnt/huanyuan/model/test_straming_wav/xiaorui_12162020_training_60_001.wav"
    sample_rate = 16000
    
    audio_data = librosa.core.load(input_wav, sr=sample_rate)[0] * 32768.0
    # audio_data = librosa.core.load(input_wav, sr=sample_rate)[0]
    # data_input = GetFrequencyFeature3(audio_data[:48000], sample_rate)
    data_input = GetFrequencyFeature3(audio_data[96000:48000+96000], sample_rate)
    print(audio_data)