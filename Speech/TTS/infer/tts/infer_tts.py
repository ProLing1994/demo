import argparse
import sys
from numba.core.types.misc import ClassDataType
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.dataset import audio
from Basic.text.mandarin.pinyin.pinyin import get_pinyin
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *

from SV.utils.infer_tools import *

from TTS.utils.tts.infer_tools import *
from TTS.utils.tts.visualizations_tools import *


def load_sv_net(cfg):
    """
    load speaker verification net
    """
    cfg_speaker_verification = load_cfg_file(cfg.speaker_verification.config_file)
    net_sv = import_network(cfg_speaker_verification, cfg.speaker_verification.model_name, cfg.speaker_verification.class_name)
    load_checkpoint(net_sv, 
                    cfg.speaker_verification.load_mode_type,
                    cfg.speaker_verification.model_dir, cfg.speaker_verification.epoch_num, cfg.speaker_verification.sub_folder_name,
                    cfg.speaker_verification.model_path,
                    cfg.speaker_verification.state_name, cfg.speaker_verification.ignore_key_list, cfg.speaker_verification.add_module_type)
    net_sv.eval().cuda()
    return cfg_speaker_verification, net_sv


def load_synthesizer_net(cfg):
    """
    load synthesizer net
    """
    cfg_synthesizer = cfg
    net_synthesizer = import_network(cfg_synthesizer, cfg.net.model_name, cfg.net.class_name)
    load_checkpoint(net_synthesizer, 
                    0,
                    cfg.general.save_dir, cfg.test.model_epoch, 'checkpoints',
                    "",
                    'state_dict', [], 0)
    net_synthesizer.eval().cuda()
    return cfg_synthesizer, net_synthesizer


class TtsSynthesizer():
    def __init__(self, config_file):
        # load config
        self.cfg = load_cfg_file(config_file)

        # load model
        # load speaker verification net
        if self.cfg.dataset.mutil_speaker and self.cfg.speaker_verification.on:
            self.cfg_speaker_verification, self.net_sv = load_sv_net(self.cfg)

        # load synthesizer net
        self.cfg_synthesizer, self.net_synthesizer = load_synthesizer_net(self.cfg)
    
    def load_speaker_info(self, speaker_id=None, speaker_wav=None):
        # load speaker_id
        self.speaker_id = speaker_id

        # load speaker_embedding
        if self.cfg.dataset.mutil_speaker and self.cfg.speaker_verification.on:
            self.speaker_embedding = embedding(self.cfg_speaker_verification, self.net_sv, speaker_wav) 
        else:
            self.speaker_embedding = None

    def synthesize(self, text):
        mel, align_score = synthesize_spectrogram(self.cfg_synthesizer, self.net_synthesizer, text, self.speaker_id, self.speaker_embedding)
        return mel, align_score


def infer(args):
    """
    模型推理
    """
    # load config
    cfg = load_cfg_file(args.config_file)

    # load text
    texts = args.text_list
    texts_name = args.text_name_list

    # load tts synthesize
    tts_synthesize = TtsSynthesizer(args.config_file)
    tts_synthesize.load_speaker_info(args.speaker_id, args.speaker_wav)

    for idx in tqdm(range(len(texts))):
        # synthesize_spectrogram
        text = texts[idx]
        mel, align_score = tts_synthesize.synthesize(text)

        ## Generating the waveform
        print("Synthesizing the waveform: ", text)
        print("Synthesizing Align Score: " , align_score)

        # save griffin lim inverted wav for debug (mel -> wav)
        wav = audio.compute_inv_mel_spectrogram(cfg, mel)

        # save 
        output_dir = os.path.join(cfg.general.save_dir, "infer")
        create_folder(output_dir)
        # wav_fpath = os.path.join(output_dir, "wave_griffin_from_mel_sample_{}_text_{}.wav".format(os.path.splitext(os.path.basename(args.speaker_wav))[0], texts_name[idx]))
        wav_fpath = os.path.join(output_dir, "wave_griffin_from_mel_spk_{}_text_{}.wav".format(args.speaker_id, texts_name[idx]))
        audio.save_wav(wav, str(wav_fpath), sampling_rate=cfg.dataset.sampling_rate)


def main():
    parser = argparse.ArgumentParser(description='Streamax TTS Training Engine')

    # # english
    # parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/tts/tts_config_english_sv2tts.py", nargs='?', help='config file')
    # parser.add_argument('-w', '--speaker_wav', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/1320_00000.mp3", nargs='?', help='config file')
    # args = parser.parse_args()
    # args.text_list = [
    #                     "activate be double you see.", 
    #                     "start recording start recording start recording start recording start recording. ",
    #                     "stop recording stop recording stop recording stop recording stop recording. ",
    #                     "mute audio mute audio mute audio mute audio mute audio. ",
    #                     "unmute audio unmute audio unmute audio unmute audio unmute audio. ",
    #                     ]
    # args.text_name_list = [
    #                         "activate_b_w_c", 
    #                         "start_recording",
    #                         "stop_recording",
    #                         "mute_audio",
    #                         "unmute_audio",
    #                         ]

    # # chinese, en
    # # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_tacotron_singlespeaker_finetune_4_3_10292021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')                      # old tacotron 单说话人，字母标签
    # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_old_tacotron_mutilspeaker_finetune_1_2_10232021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')                     # old tacotron 多说话人，字母标签
    # parser.add_argument('-w', '--speaker_wav', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00050001.wav", nargs='?', help='config file')
    # # parser.add_argument('-w', '--speaker_wav', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00730005.wav", nargs='?', help='config file')
    # args = parser.parse_args()
    # args.text_list = [
    #                     # " ".join(get_pinyin("道路千万条安全第一条")),
    #                     # " ".join(get_pinyin("时刻牢记以人为本安全第一的原则。")),
    #                     " ".join(get_pinyin("道路千万条，安全第一条。时刻牢记以人为本，安全第一的原则。")),
    #                     # " ".join(get_pinyin("今天星期五天气好真开心")),
    #                     # " ".join(get_pinyin("今天星期五，天气好，真开心。")),
    #                     ]
    # args.text_name_list = [
    #                         # "道路千万条安全第一条",
    #                         # "时刻牢记以人为本安全第一的原则。",
    #                         "道路千万条，安全第一条。时刻牢记以人为本，安全第一的原则。",
    #                         # "今天星期五天气好真开心",
    #                         # "今天星期五，天气好，真开心。",
    #                         ]

    # chinese, prosody py
    # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_singlespeaker_prosody_py_1_0_11102021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')           # tacotron2 单说话人，韵律标签
    # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_singlespeaker_prosody_py_1_1_diff_feature_11292021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')  # tacotron2 单说话人，韵律标签，与 vocoder 采用相同的特征
    # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_singlespeaker_prosody_py_1_2_diff_feature_11292021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')  # tacotron2 单说话人，韵律标签，与 vocoder 采用相同的特征，stop 预测位置优化
    parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_singlespeaker_prosody_py_1_3_diff_feature_11292021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')  # tacotron2 单说话人，韵律标签，与 vocoder 采用相同的特征，stop 预测位置优化，静音处优化
    # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_mutilspeaker_prosody_py_1_2_11102021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')              # tacotron2 多说话人 speaker_id_embedding，韵律标签，标签不做修改
    # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_mutilspeaker_prosody_py_1_3_11102021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')              # tacotron2 多说话人 speaker_id_embedding，韵律标签，标签不做修改
    parser.add_argument('-s', '--speaker_id', type=int, default=20, nargs='?', help='config file')
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