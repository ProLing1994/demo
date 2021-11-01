import argparse
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.dataset import audio
from Basic.utils.train_tools import *


def test_griffin(args):
    # load config
    cfg = load_cfg_file(args.config_file)

    wav = audio.load_wav(args.wav_file, cfg.dataset.sample_rate)
    spec = audio.compute_mel_spectrogram(cfg, wav).T
    wav = audio.compute_inv_mel_spectrogram(cfg, spec)
    wav_fpath = os.path.join(args.output_path, "wave_griffin_from_mel_{}.wav".format(os.path.splitext(os.path.basename(args.wav_file))[0]))
    audio.save_wav(wav, str(wav_fpath), sr=cfg.dataset.sample_rate)


def main():
    parser = argparse.ArgumentParser(description='Streamax SV2TTS Test Engine')
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/sv2tts/tts_config_chinese_sv2tts.py", nargs='?', help='config file')
    # parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00050001.wav", nargs='?', help='config file')
    parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00730005.wav", nargs='?', help='config file')
    parser.add_argument('--output_path', type=str, default="/home/huanyuan/temp", nargs='?', help='config file')

    args = parser.parse_args()
    test_griffin(args)


if __name__ == "__main__":
    main()