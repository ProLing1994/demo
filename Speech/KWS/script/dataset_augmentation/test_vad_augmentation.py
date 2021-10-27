import librosa
import numpy as np
import os 
import sys
import struct
import webrtcvad

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio
from Basic.config import hparams


def main():
    input_wav = "/home/huanyuan/temp/RM_KWS_XIAOAN_xiaoan_S068M0D42T124.wav"
    output_dir = "/home/huanyuan/temp"
    sample_rate = 8000 

    # window size of the vad. 
    # Must be either 10, 20 or 30 milliseconds. This sets the granularity of the VAD. Should not need to be changed.
    vad_window_length_list = [10, 20, 30]
    
    # vad mode
    # 0: Normal，1：low Bitrate，2：Aggressive，3：Very Aggressive
    vad_mode_list = [0, 1, 2, 3]

    for vad_window_length_idx in range(len(vad_window_length_list)):
        vad_chunk_size = (vad_window_length_list[vad_window_length_idx] * sample_rate) // 1000

        for vad_mode_idx in range(len(vad_mode_list)):
            vad_mode = vad_mode_list[vad_mode_idx]

            data = audio.load_wav(input_wav, sample_rate)
            # Trim the end of the audio to have a multiple of the window size
            data = data[:len(data) - (len(data) % vad_chunk_size)]
            
            # Convert the float waveform to 16-bit mono PCM
            pcm_wave = struct.pack("%dh" % len(data), *(np.round(data * hparams.int16_max)).astype(np.int16))
            
            # Perform voice activation detection
            vad = webrtcvad.Vad(mode=vad_mode)
            vad_flags = []
            for window_start in range(0, len(data), vad_chunk_size):
                window_end = window_start + vad_chunk_size
                vad_flags.append(vad.is_speech(pcm_wave[window_start * 2 : window_end * 2],
                                                sample_rate=sample_rate))
            
            for vad_idx in range(len(vad_flags)):
                if vad_flags[vad_idx] == 0:
                    window_start = vad_idx * vad_chunk_size
                    window_end = window_start + vad_chunk_size
                    data[window_start: window_end] = 0

            output_path = os.path.join(output_dir, os.path.basename(input_wav).split('.')[0] + "_vad_chunk_size_{}_vad_mode_{}.wav".format(vad_chunk_size, vad_mode))
            audio.save_wav(data.copy(), output_path, sample_rate)

if __name__ == "__main__":
    main()