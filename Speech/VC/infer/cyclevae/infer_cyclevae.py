import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *
from VC.utils.cyclevae.infer_tools import VcCycleVaeInfer


def infer(args):
    """
    模型推理
    """
    # load config
    cfg = load_cfg_file(args.config_file)

    # load cyclevae
    vc_cyclevae_infer = VcCycleVaeInfer(cfg)

    for speaker_trg in args.speaker_trg_list:
        
        # load spk info
        vc_cyclevae_infer.load_spk_info(args.speaker_src, speaker_trg)
        
        wav_list = os.listdir(args.wav_folder)
        wav_list.sort()
        for idx in tqdm(range(len(wav_list)), speaker_trg):
            
            wav_path = os.path.join(args.wav_folder, wav_list[idx])
            wav_name = wav_list[idx].split('.')[0]
            wav_cv, wav_anasyn = vc_cyclevae_infer.voice_conversion(wav_path)

            # Save it on the disk
            output_dir = os.path.join(cfg.general.save_dir, "infer")
            create_folder(output_dir)

            wav_fpath = os.path.join(output_dir, "wav_cyclevae_{}_{}.wav".format(speaker_trg, wav_name))
            audio.save_wav(wav_cv, str(wav_fpath), sampling_rate=cfg.dataset.sampling_rate)
            
            wav_fpath = os.path.join(output_dir, "wav_{}_{}.wav".format(args.speaker_src, wav_name))
            audio.save_wav(wav_anasyn, str(wav_fpath), sampling_rate=cfg.dataset.sampling_rate)

    return 


def main(): 
    parser = argparse.ArgumentParser(description='Streamax VC Infer Engine')
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/VC/config/cyclevae/vc_config_cyclevae.py", nargs='?', help='config file')
    args = parser.parse_args()

    args.speaker_src = "SEF1"
    # args.speaker_trg_list = ["SEM1"]
    args.speaker_trg_list = ["SEF1", "SEM1", "SEF2", "SEM2"]
    args.wav_folder = "/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/dataset/test/SEF1/"

    infer(args)


if __name__ == "__main__":
    main()