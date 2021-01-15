import argparse
import cv2
import os
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/ASR')
from impl.asr_data_loader_cimpl import WavLoader
from impl.asr_feature_cimpl import Feature

def demo(args):

    # init 
    n_fft = 512
    feature_freq = 48
    time_seg_ms = 32
    time_step_ms = 10

    window_size_ms = 3000
    window_stride_ms = 2000
    sample_rate = 16000
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    wave_list = os.listdir(args.audio_folder)
    for idx in range(len(wave_list)):

        wave_path = os.path.join(args.audio_folder, wave_list[idx])
        print("[Information:] Audio path: ", wave_path)
        
        # load wave
        wave_data = WavLoader.ReadWave(wave_path)

        # forward
        windows_times = int((len(wave_data) - window_size_samples) * 1.0 / window_stride_samples)
        for times in range(windows_times):
            print("\n[Information:] Wave start time: ", times * window_stride_samples)

            audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]
            feature_data = Feature.GetMelIntFeature(audio_data, len(audio_data), n_fft, sample_rate, time_seg_ms, time_step_ms, feature_freq)

            # cv2.imshow("feature_data",feature_data)
            # cv2.waitKey() 
            # print(feature_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streamax KWS Training Engine')
    parser.add_argument('-i', '--audio_folder', type=str, default="/home/huanyuan/share/audio_data/")
    args = parser.parse_args()

    demo(args)