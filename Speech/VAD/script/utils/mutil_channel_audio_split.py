import argparse
import numpy as np
import os
import pyaudio
from tqdm import tqdm
import wave


def write_wave(wave_frames, wave_output_path, framerate=16000):
    """
    :param wave_frames: 是二进制的数据
    :param wave_output_path: 输出的位置
    :return: 
    """
    p = pyaudio.PyAudio()
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    RATE = framerate  # 这个要跟原音频文件的比特率相同
    wf = wave.open(wave_output_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(wave_frames)
    wf.close()


def mutil_channel_audio_split(args):
    file_list = os.listdir(args.input_folder)
    file_list.sort()

    # mkdir
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for idx in tqdm(range(len(file_list))):
        file_name = file_list[idx]

        if not file_name.endswith('.wav'):
            continue

        # find name_list
        bool_find_name_list = False
        for name in args.name_list:
            if name in file_name:
                bool_find_name_list = True
        if bool_find_name_list:
            continue

        audio_path = os.path.join(args.input_folder, file_name)

        f = wave.open(audio_path, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)
        f.close()
        
        # check 
        assert nchannels == len(args.name_list)
        assert framerate == args.sample_rate

        # get wave str_data
        wave_data = np.fromstring(str_data, dtype=np.int16)
        wave_data.shape = -1, nchannels
        wave_data = wave_data.T

        for idx in range(nchannels):
            wave_frames = wave_data[idx].tostring()
            wave_output_path = os.path.join(args.output_folder, file_name.split('.')[0] + '_' + args.name_list[idx] + '.wav')

            # 语音不存在，则不生成对应文件
            print(file_name.split('.')[0], args.name_list[idx], wave_data[idx].max(), wave_data[idx].min())
            if abs(wave_data[idx].max()) == abs(wave_data[idx].min()) and abs(wave_data[idx].max()) == 1:
                continue
            
            write_wave(wave_frames, wave_output_path, framerate=args.sample_rate)

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/实车场景/"
    args.output_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/实车场景/处理音频/"
    args.sample_rate = 16000
    args.name_list = ['adpro', 'mic', 'danbin_ori', 'danbin_asr']

    mutil_channel_audio_split(args)

if __name__ == '__main__':
    main() 