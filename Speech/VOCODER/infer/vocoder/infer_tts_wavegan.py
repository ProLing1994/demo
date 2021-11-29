import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.dataset import audio
from Basic.text.mandarin.pinyin.pinyin import get_pinyin
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *

from SV.utils.infer_tools import *

from TTS.infer.tts.infer import TtsSynthesizer
from TTS.utils.tts.infer_tools import *
from TTS.utils.tts.visualizations_tools import *

from VOCODER.utils.vocoder.infer_tools import *


def load_vocoder_net(cfg):
    """
    load vocoder net 
    """
    # load vocoder net 
    cfg_vocoder = cfg
    net_vocoder = import_network(cfg_vocoder, cfg.net.generator_model_name, cfg.net.generator_class_name)

    state_dict = load_state_dict(cfg.general.save_dir, cfg.test.model_epoch, 'checkpoints')
    net_vocoder.load_state_dict(state_dict["model"]["generator"])

    # load stats
    dataset_name = '_'.join(cfg.general.dataset_list)
    dataset_audio_normalize_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_normalize_hdf5')
    stats = os.path.join(dataset_audio_normalize_dir, dataset_name +  "_stats.h5")
    if isinstance(net_vocoder, torch.nn.parallel.DataParallel):
        net_vocoder.module.register_stats(stats)
    else:
        net_vocoder.register_stats(stats)

    # add pqmf if needed
    if cfg.net.yaml["generator_params"]["out_channels"] > 1:
        raise NotImplementedError

    # ???
    if isinstance(net_vocoder, torch.nn.parallel.DataParallel):
        net_vocoder.module.remove_weight_norm()
    else:
        net_vocoder.remove_weight_norm()

    net_vocoder.eval().cuda()
    return cfg_vocoder, net_vocoder


class TtsVocoder():
    def __init__(self, config_file):
        # load config
        self.cfg = load_cfg_file(config_file)

        # load vocoder net 
        self.cfg_vocoder, self.net_vocoder = load_vocoder_net(self.cfg)

    def synthesize(self, mel):
        wav = infer_wavegan(self.cfg, self.net_vocoder, mel, normalize_before=self.cfg.dataset.normalize_bool)
        return wav


def infer(args):
    """
    模型推理，通过滑窗的方式得到每一小段 embedding，随后计算 EER
    """
    # load config
    cfg = load_cfg_file(args.vocoder_config_file)

    # load text
    texts = args.text_list
    texts_name = args.text_name_list

    # load tts synthesize
    tts_synthesize = TtsSynthesizer(args.tts_config_file)
    tts_synthesize.load_speaker_info(args.speaker_id, args.speaker_wav)

    # load tts vocoder
    tts_vocoder = TtsVocoder(args.vocoder_config_file)

    for idx in tqdm(range(len(texts))):
        # synthesize_spectrogram
        text = texts[idx]
        mel, align_score = tts_synthesize.synthesize(text)

        ## Generating the waveform
        print("Synthesizing the waveform: ", text)
        print("Synthesizing Align Score: " , align_score)

        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        wav = tts_vocoder.synthesize(mel)
        
        ## Post-generation
        # Trim excess silences to compensate for gaps in spectrograms
        wav = audio.preprocess_wav(wav, cfg.dataset.sampling_rate)

        # Save it on the disk
        output_dir = os.path.join(cfg.general.save_dir, "infer")
        create_folder(output_dir)
        wav_fpath = os.path.join(output_dir, "wave_wavegan_from_mel_spk_{}_text_{}.wav".format(args.speaker_id, texts_name[idx]))
        audio.save_wav(wav, str(wav_fpath), sampling_rate=cfg.dataset.sampling_rate)


def main():
    parser = argparse.ArgumentParser(description='Streamax TTS Training Engine')

    # chinese
    parser.add_argument('-t', '--tts_config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_singlespeaker_prosody_py_1_0_11102021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')           # tacotron2 单说话人，韵律标签
    # parser.add_argument('-t', '--tts_config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_mutilspeaker_prosody_py_1_3_11102021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')              # tacotron2 多说话人 speaker_id_embedding，韵律标签，标签不做修改
    # parser.add_argument('-v', '--vocoder_config_file', type=str, default="/mnt/huanyuan2/model/tts_vocoder/chinese_tts_vocoder/wavegan_chinese_singlespeaker_1_0_11232021/vocoder_config_chinese_wavegan.py", nargs='?', help='config file')            # ParallelWaveGANGenerator, fft_size=800, preemphasis
    parser.add_argument('-v', '--vocoder_config_file', type=str, default="/mnt/huanyuan2/model/tts_vocoder/chinese_tts_vocoder/wavegan_chinese_singlespeaker_1_1_normalize_11232021/vocoder_config_chinese_wavegan.py", nargs='?', help='config file')  # ParallelWaveGANGenerator, fft_size=800, preemphasis, nomalize
    parser.add_argument('-s', '--speaker_id', type=int, default=0, nargs='?', help='config file')
    parser.add_argument('-w', '--speaker_wav', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00050001.wav", nargs='?', help='config file')
    args = parser.parse_args()
    args.text_list = [
                        "ka2-er2-pu3 / pei2-wai4-sun1 wan2-hua2-ti1.", 
                        "dao4-lu4 / qian1-wan4-tiao2, an1-quan2 / di4-yi1-tiao2.",
                        "shi2-ke4 lao2-ji4 / yi3-ren2-wei2-ben3, an1-quan2-di4-yi1-de5 yuan2-ze2.",
                        "dao4-lu4 / qian1-wan4-tiao2, an1-quan2 / di4-yi1-tiao2, shi2-ke4 lao2-ji4 / yi3-ren2-wei2-ben3, an1-quan2-di4-yi1-de5 yuan2-ze2.",
                        "jin1-tian1 / xing1-qi1-wu3, tian1-qi4-hao3, zhen1 kai1-xin1.",
                        ]
    args.text_name_list = [
                            "卡尔普陪外孙玩滑梯。",
                            "道路千万条，安全第一条",
                            "时刻牢记以人为本，安全第一的原则。",
                            "道路千万条，安全第一条。时刻牢记以人为本，安全第一的原则。",
                            "今天星期五，天气好，真开心。",
                            ]

    infer(args)


if __name__ == "__main__":
    main()