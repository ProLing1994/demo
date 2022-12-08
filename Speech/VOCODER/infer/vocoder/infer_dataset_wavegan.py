import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.dataset import audio
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *

from VOCODER.infer.vocoder.infer_tts_wavegan import TtsVocoder
from VOCODER.dataset.vocoder.vocoder_wavegan_dataset_preload_audio_hdf5 import VocoderWaveGanDataset
from VOCODER.dataset.vocoder.vocoder_wavegan_vc_dataset_preload_audio_hdf5 import VocoderWaveGanVcDataset

def infer(args):
    """
    模型推理
    """
    # load config
    cfg = load_cfg_file(args.vocoder_config_file)

    # load tts vocoder
    tts_vocoder = TtsVocoder(args.vocoder_config_file)

    # load dataset
    # define training dataset and testing dataset
    if cfg.dataset.compute_mel_type == "world":
        dataset = VocoderWaveGanVcDataset(cfg, hparams.TRAINING_NAME, bool_return_name=True)
    elif cfg.dataset.compute_mel_type == "fbank_nopreemphasis_log_manual":
        dataset = VocoderWaveGanDataset(cfg, hparams.TRAINING_NAME, bool_return_name=True)
    else:
        raise NotImplementedError

    print("The number of files: {}".format(len(dataset)))

    for _, mel, data_name in tqdm(dataset):
        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        wav_synthesize = tts_vocoder.synthesize_no_normalize(mel.T)
        
        ## Post-generation
        # Trim excess silences to compensate for gaps in spectrograms
        # wav_synthesize = audio.preprocess_wav(wav_synthesize, cfg.dataset.sampling_rate)
        wav_synthesize = audio.preprocess_wav(wav_synthesize, cfg.dataset.sampling_rate, bool_trim_silence=False)

        # Save it on the disk
        output_dir = os.path.join(cfg.general.save_dir, "wavs_test")
        create_folder(output_dir)
        wav_fpath = os.path.join(output_dir, data_name)
        audio.save_wav(wav_synthesize, str(wav_fpath), sampling_rate=cfg.dataset.sampling_rate)

def main():
    parser = argparse.ArgumentParser(description='Streamax TTS Training Engine')

    # # fbank 
    # # chinese
    # parser.add_argument('-v', '--vocoder_config_file', type=str, default="/mnt/huanyuan/model/tts_vocoder/chinese_tts_vocoder/wavegan_chinese_singlespeaker_1_2_normalize_diff_feature_11292021/vocoder_config_chinese_wavegan.py", nargs='?', help='config file')  # ParallelWaveGANGenerator, fft_size=1024, nomalize
    
    # world
    # english
    parser.add_argument('-v', '--vocoder_config_file', type=str, default="/mnt/huanyuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_0_normalize_world_01042022/vc_vocoder_config_english_wavegan.py", nargs='?', help='config file')  # ParallelWaveGANGenerator, vc world feature
    args = parser.parse_args()

    infer(args)


if __name__ == "__main__":
    main()