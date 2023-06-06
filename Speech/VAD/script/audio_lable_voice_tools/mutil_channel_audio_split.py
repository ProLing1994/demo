import argparse
import numpy as np
import os
import pyaudio
from tqdm import tqdm
import wave
import shutil


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

    # mkdir
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # file_list
    file_list = os.listdir(args.input_folder)
    file_list.sort()

    for idx in tqdm(range(len(file_list))):
        file_name = file_list[idx]

        # txt 文件，执行拷贝
        if file_name.endswith('.txt'):
            input_txt_path = os.path.join(args.input_folder, file_name)
            output_txt_path = os.path.join(args.output_folder, file_name)
            shutil.copy(input_txt_path, output_txt_path)
        
        # wav 文件，多通道音频转单通道音频
        if file_name.endswith('.wav'):
            audio_path = os.path.join(args.input_folder, file_name)

            # 加载音频（多通道）
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
                
                # 保存音频
                write_wave(wave_frames, wave_output_path, framerate=args.sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/collect/20230605/temp/")
    parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/collect/20230605/test_out/")
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--name_list', type=str, default="mic_130cm,phone,adplus1_0_normal,adplus1_0_70cm,adplus1_0_100cm,adplus2_0_normal,adplus2_0_70cm,adplus2_0_100cm")
    args = parser.parse_args()

    args.name_list = args.name_list.split(',')

    mutil_channel_audio_split(args)