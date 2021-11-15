import argparse
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.dataset import audio
from Basic.text.mandarin.pinyin.pinyin import get_pinyin
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *

from SV.utils.infer_tools import *

from TTS.utils.sv2tts.infer_tools import *
from TTS.utils.sv2tts.visualizations_tools import *


def infer(args):
    """
    模型推理，通过滑窗的方式得到每一小段 embedding，随后计算 EER
    """
    # load config
    cfg = load_cfg_file(args.config_file)

    # load speaker verification net
    if cfg.general.mutil_speaker:
        cfg_speaker_verification = load_cfg_file(cfg.speaker_verification.config_file)
        sv_net = import_network(cfg_speaker_verification, 
                                cfg.speaker_verification.model_name, 
                                cfg.speaker_verification.class_name)
        load_checkpoint(sv_net, 
                        cfg.general.load_mode_type,
                        cfg.speaker_verification.model_dir, cfg.speaker_verification.epoch, cfg.general.finetune_sub_folder_name,
                        cfg.general.finetune_model_path,
                        cfg.general.finetune_state_name, cfg.general.finetune_ignore_key_list, cfg.general.finetune_add_module_type)
        sv_net.eval()

    # load synthesizer net
    cfg_synthesizer = cfg
    net_synthesizer = import_network(cfg_synthesizer, 
                                    cfg.net.model_name, 
                                    cfg.net.class_name)
    load_checkpoint(net_synthesizer, 
                    cfg.general.load_mode_type,
                    cfg.general.save_dir, cfg.test.model_epoch, cfg.general.finetune_sub_folder_name,
                    cfg.general.finetune_model_path,
                    cfg.general.finetune_state_name, cfg.general.finetune_ignore_key_list, cfg.general.finetune_add_module_type)
    net_synthesizer.eval()

    # load text
    texts = args.text_list
    texts_name = args.text_name_list

    # load embedding
    if cfg.general.mutil_speaker:
        embed = embedding(cfg_speaker_verification, sv_net, args.wav_file) 
        embeds = [embed] * len(args.text_list)
    else:
        embeds = None

    # synthesize_spectrograms
    specs = synthesize_spectrograms(cfg_synthesizer, net_synthesizer, texts, embeds)
    
    for spec_idx in range(len(specs)):

        spec = specs[spec_idx]

        ## Generating the waveform
        print("Synthesizing the waveform: ", texts[spec_idx])

        # save griffin lim inverted wav for debug (mel -> wav)
        wav = audio.compute_inv_mel_spectrogram(cfg, spec)
        output_dir = os.path.join(cfg.general.save_dir, "infer")
        create_folder(output_dir)
        wav_fpath = os.path.join(output_dir, "wave_griffin_from_mel_sample_{}_text_{}.wav".format(os.path.splitext(os.path.basename(args.wav_file))[0], texts_name[spec_idx]))
        audio.save_wav(wav, str(wav_fpath), sr=cfg.dataset.sample_rate)


def main():
    parser = argparse.ArgumentParser(description='Streamax SV2TTS Training Engine')

    # # english
    # parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/sv2tts/tts_config_english_sv2tts.py", nargs='?', help='config file')
    # parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/1320_00000.mp3", nargs='?', help='config file')
    # args = parser.parse_args()
    # args.text_list = ["activate be double you see."]
    # args.text_name_list = ["activate_b_w_c"]

    # # chinese, en
    # # parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/sv2tts/tts_config_chinese_sv2tts.py", nargs='?', help='config file')
    # # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_finetune_1_2_10232021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_tacotron_singlespeaker_finetune_4_3_10292021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')
    # parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00050001.wav", nargs='?', help='config file')
    # # parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00730005.wav", nargs='?', help='config file')
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
    parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_singlespeaker_prosody_py_1_0_11102021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')
    parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00050001.wav", nargs='?', help='config file')
    args = parser.parse_args()
    args.text_list = [
                        "dao4-lu4 / qian1-wan4-tiao2, an1-quan2 / di4-yi1-tiao2. ",
                        # " ".join(get_pinyin("时刻牢记以人为本安全第一的原则。")),
                        # " ".join(get_pinyin("道路千万条，安全第一条。时刻牢记以人为本，安全第一的原则。")),
                        # " ".join(get_pinyin("今天星期五天气好真开心")),
                        # " ".join(get_pinyin("今天星期五，天气好，真开心。")),
                        ]
    args.text_name_list = [
                            "道路千万条安全第一条",
                            # "时刻牢记以人为本安全第一的原则。",
                            # "道路千万条，安全第一条。时刻牢记以人为本，安全第一的原则。",
                            # "今天星期五天气好真开心",
                            # "今天星期五，天气好，真开心。",
                            ]

    infer(args)


if __name__ == "__main__":
    main()