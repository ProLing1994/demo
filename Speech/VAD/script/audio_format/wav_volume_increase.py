import argparse
import os
import soundfile as sf

from tqdm import tqdm

def get_sub_filepaths_suffix(folder, suffix='.wav'):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if not name.endswith(suffix):
                continue
            path = os.path.join(root, name)
            paths.append(path)
    return paths

def volume_increase(args):
    # init 
    wave_list = get_sub_filepaths_suffix(args.input_dir)
    for idx in tqdm(range(len(wave_list))):
        audio_path = wave_list[idx]
        wav, source_sr = sf.read(audio_path)
        wav = wav * args.volume_increase_scale 

        output_path = str(audio_path).split('.')[0] + "volume_increase_{}.wav".format(args.volume_increase_scale)
        sf.write(output_path, wav, source_sr)

def main():
    parser = argparse.ArgumentParser(description="Sudio Format")
    args = parser.parse_args()
    args.input_dir = "/home/huanyuan/share/audio_data/weakup/weakup_xiaoan8k/adpro2_1_录制声音较小音频/"
    args.volume_increase_scale = 10

    volume_increase(args)
    

if __name__ == "__main__":
    main()