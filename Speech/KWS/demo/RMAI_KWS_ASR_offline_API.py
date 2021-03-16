import argparse
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_data_loader_pyimpl import WaveLoader
from ASR.impl.asr_feature_pyimpl import Feature

def KWS_ASR_offine():
    # init 
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    # load wave
    wave_loader = WaveLoader(sample_rate)
    wave_loader.load_data(args.input_wav)
    wave_data = wave_loader.to_numpy()

    # sliding window
    windows_times = int((len(wave_data) - window_size_samples) * 1.0 / window_stride_samples) + 1
    for times in range(windows_times):

        # get audio data
        audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]
        print("audio data stram: {} - {}, length: {} ".format(times * int(window_stride_samples), times * int(window_stride_samples) + int(window_size_samples), len(audio_data)))

if __name__ == "__main__":
    # params
    sample_rate = 16000
    window_size_ms = 1000
    window_stride_ms = 1000

    # argparse
    default_input_wav = "/mnt/huanyuan/model/test_straming_wav/xiaorui_12162020_training_60_001.wav"

    parser = argparse.ArgumentParser(description='Streamax KWS ASR offine Engine')
    parser.add_argument('--input_wav', type=str, default=default_input_wav)
    args = parser.parse_args()

    KWS_ASR_offine()