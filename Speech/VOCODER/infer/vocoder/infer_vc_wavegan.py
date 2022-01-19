import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.dataset import audio
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *

from VC.utils.cyclevae.infer_tools import VcCycleVaeInfer

from VOCODER.infer.vocoder.infer_tts_wavegan import TtsVocoder


def infer(args):
    """
    模型推理
    """
    # load config
    cfg = load_cfg_file(args.vocoder_config_file)

    # load cyclevae
    cfg_vc = load_cfg_file(args.vc_config_file)
    vc_cyclevae_infer = VcCycleVaeInfer(cfg_vc)

    # load tts vocoder
    tts_vocoder = TtsVocoder(args.vocoder_config_file)

    for speaker_trg in args.speaker_trg_list:
        # load spk info
        vc_cyclevae_infer.load_spk_info(args.speaker_src, speaker_trg)

        wav_list = os.listdir(args.wav_folder)
        wav_list.sort()

        for idx in tqdm(range(len(wav_list)), speaker_trg):

            wav_path = os.path.join(args.wav_folder, wav_list[idx])
            wav_name = wav_list[idx].split('.')[0]
            _, feat_cv, _ = vc_cyclevae_infer.voice_conversion(wav_path)

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            wav = tts_vocoder.synthesize(feat_cv.T)

            # Save it on the disk
            output_dir = os.path.join(cfg.general.save_dir, "infer")
            create_folder(output_dir)
            wav_fpath = os.path.join(output_dir,"wav_cyclevae_{}2{}_{}.wav".format(args.speaker_src, speaker_trg, wav_name))
            audio.save_wav(wav, str(wav_fpath), sampling_rate=cfg.dataset.sampling_rate)


def main():
    parser = argparse.ArgumentParser(description='Streamax VC VOCODER Infer Engine')

    # english
    parser.add_argument('-c', '--vc_config_file', type=str, default="/mnt/huanyuan/model/vc/english_vc/vc_english_cyclevae_world_1_1_12302021/config.py", nargs='?', help='config file')              # vc 声音转换，world 特征
    # parser.add_argument('-v', '--vocoder_config_file', type=str, default="/mnt/huanyuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_0_normalize_world_01042022/vc_vocoder_config_english_wavegan.py", nargs='?', help='config file')  # ParallelWaveGANGenerator, vc world feature（限定多说话人转移，效果还行）
    # parser.add_argument('-v', '--vocoder_config_file', type=str, default="/mnt/huanyuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_1_normalize_world_cyclevae_reconst_01112022/vc_vocoder_config_english_wavegan.py", nargs='?', help='config file')  # ParallelWaveGANGenerator, vc world feature, vc finetune（限定多说话人转移，效果还行）
    parser.add_argument('-v', '--vocoder_config_file', type=str, default="/mnt/huanyuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_2_normalize_world_cyclevae_reconst_01112022/vc_vocoder_config_english_wavegan.py", nargs='?', help='config file')  # ParallelWaveGANGenerator, vc world feature, vc finetune（限定多说话人转移，效果还行）
    args = parser.parse_args()

    args.speaker_src = "SEF1"
    args.speaker_trg_list = ["SEF1", "SEM1", "SEF2", "SEM2"]
    args.wav_folder = "/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/dataset/test/SEF1/"

    # # chinese
    # parser.add_argument('-c', '--vc_config_file', type=str, default="/mnt/huanyuan/model/vc/english_vc/vc_english_cyclevae_world_1_1_12302021/config.py", nargs='?', help='config file')              # vc 声音转换，world 特征
    # parser.add_argument('-v', '--vocoder_config_file', type=str, default="/mnt/huanyuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_0_normalize_world_01042022/vc_vocoder_config_english_wavegan.py", nargs='?', help='config file')  # ParallelWaveGANGenerator, vc world feature（限定多说话人转移，效果还行）
    # args = parser.parse_args()

    # args.speaker_src = "SEF1"
    # args.speaker_trg_list = ["SEF1", "SEM1", "SEF2", "SEM2"]
    # args.wav_folder = "/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/dataset/test/SEF1/"

    infer(args)


if __name__ == "__main__":
    main()