import argparse
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_data_loader_pyimpl import WaveLoader_Librosa as WaveLoader

def find_audio():
    # mkdir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # init 
    wave_list = os.listdir(args.input_dir)

    for idx in tqdm(range(len(wave_list))):
        wave_name = wave_list[idx]
        wave_path = os.path.join(args.input_dir, wave_list[idx])

        if args.file_format in wave_name:
            output_path = os.path.join(args.output_dir, wave_name.split('.')[0] + '.wav')
            print(wave_path, '->', output_path)
            # shutil.copy(wave_path, output_path)
            shutil.move(wave_path, output_path)


if __name__ == "__main__":
    default_input_dir = "/mnt/huanyuan/data/speech/Recording/RM_Activatebwc/office/danbing_16k/"
    default_output_dir = "/mnt/huanyuan/data/speech/Recording/RM_Activatebwc/office/danbing_ori_16k/"
    
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    args = parser.parse_args()

    # D0_ 单兵_ori   D1_ Jabra桌面录音设备   D2_ ADkit    D3_ c6dai    D4_ ADplus/ADpro    D5_ 手机    D6_ 多阵列mic板    D7_ 单兵_asr
    args.file_format = "D0"
    find_audio()