import argparse
import os

import impl.asr_data_loader_pyimpl as WaveLoader
from impl.file_tools import load_module_from_disk
from RMAI_KWS_ASR_API import KwsAsrApi

def KWS_ASR_offine():
    # init 
    kws_asr_api = KwsAsrApi(bool_do_kws_weakup=True, bool_do_asr=True, bool_gpu=True)

    # load wave
    wave_loader = WaveLoader.WaveLoader_Librosa(kws_asr_api.sample_rate())
    wave_loader.load_data(args.input_wav)
    wave_data = wave_loader.to_numpy()
    
    # sliding window
    windows_times = int((len(wave_data) - kws_asr_api.window_size_samples()) * 1.0 / kws_asr_api.window_stride_samples()) + 1
    for times in range(windows_times):
    
        # get audio data
        audio_data = wave_data[times * int(kws_asr_api.window_stride_samples()): times * int(kws_asr_api.window_stride_samples()) + int(kws_asr_api.window_size_samples())]
        print("[Information:] Audio data stream: {} - {}, length: {} ".format((times * int(kws_asr_api.window_stride_samples())), (times * int(kws_asr_api.window_stride_samples()) + int(kws_asr_api.window_size_samples())), len(audio_data)))

        output_str = kws_asr_api.run_kws_asr(audio_data)
        # print("[Information:] result: {}".format(output_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streamax KWS ASR offine Engine')
    parser.add_argument('--input_wav', type=str, default="/home/huanyuan/code/demo/Speech/API/Kws_weakup_Asr/audio/test-kws-xiaorui-asr-mandarin-taxi_001.wav")
    args = parser.parse_args()

    KWS_ASR_offine()