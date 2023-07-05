import sys

from RMAI_KWS_ASR_API import KwsAsrApi

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# import ASR.impl.asr_data_loader_cimpl as WaveLoader_C
import ASR.impl.asr_data_loader_pyimpl as WaveLoader_Python


# options 
## English
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_BWC_bpe.py"
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_BWC_phoneme.py"
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_BWC_bpe_phoneme.py"

## Chinese
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_MANDARIN_TAXI_16k_64dim.py"
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_MANDARIN_TAXI_8k_56dim.py"
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_MTA_XIAOAN.py"
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_XIAORUI.py"
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_CQ_TAXI_3s.py"
cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_MTA_GORILA.py"


def KWS_ASR_offine():
    # init 
    # kws_asr_api = KwsAsrApi(cfg_path = cfg_path, bool_do_kws_weakup=True, bool_do_asr=True, bool_gpu=True)
    # kws_asr_api = KwsAsrApi(cfg_path = cfg_path, bool_do_kws_weakup=False, bool_do_asr=True, bool_gpu=True)
    kws_asr_api = KwsAsrApi(cfg_path = cfg_path, bool_do_kws_weakup=True, bool_do_asr=False, bool_gpu=True)
    # kws_asr_api = KwsAsrApi(cfg_path = cfg_path, bool_do_kws_weakup=True, bool_do_asr=False, bool_do_sv=True, bool_gpu=True)

    # load wave
    # wave_loader = WaveLoader_C.WaveLoader(kws_asr_api.sample_rate())
    # wave_loader = WaveLoader_Python.WaveLoader_Soundfile(kws_asr_api.sample_rate())
    wave_loader = WaveLoader_Python.WaveLoader_Librosa(kws_asr_api.sample_rate())
    wave_loader.load_data(kws_asr_api.input_wav())
    wave_data = wave_loader.to_numpy()

    # sliding window
    windows_times = int((len(wave_data) - kws_asr_api.window_size_samples()) * 1.0 / kws_asr_api.window_stride_samples()) + 1
    for times in range(windows_times):

        # get audio data
        audio_data = wave_data[times * int(kws_asr_api.window_stride_samples()): times * int(kws_asr_api.window_stride_samples()) + int(kws_asr_api.window_size_samples())]
        print("[Information:] Audio data stream: {} - {}, length: {} ".format((times * int(kws_asr_api.window_stride_samples())), (times * int(kws_asr_api.window_stride_samples()) + int(kws_asr_api.window_size_samples())), len(audio_data)))

        kws_asr_api.run_kws_asr(audio_data)


if __name__ == "__main__":
    # 实现功能：语音唤醒 wakeup 和关键词检索 asr 共同工作，目的是共用一套特征，节约资源
    # 方案一：实现 wakeup + asr 
    # 方案二：在无 wakeup 的情况下，实现 asr
    KWS_ASR_offine()       