import argparse
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.dataset import audio
from Basic.text.mandarin.pinyin import get_pinyin
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
                        cfg.speaker_verification.epoch, 
                        cfg.speaker_verification.model_dir)
        sv_net.eval()

    # load synthesizer net
    cfg_synthesizer = cfg
    net_synthesizer = import_network(cfg_synthesizer, 
                                    cfg.net.model_name, 
                                    cfg.net.class_name)
    load_checkpoint(net_synthesizer, 
                    cfg.test.model_epoch, 
                    cfg.general.save_dir)
    net_synthesizer.eval()

    # load embedding
    if cfg.general.mutil_speaker:
        embed = embedding(cfg_speaker_verification, sv_net, args.wav_file) 
        embeds = [embed]
    else:
        embeds = None

    # load text
    texts = [args.text]
    texts_name = [args.text_name]

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
        wav_fpath = os.path.join(output_dir, "wave_from_mel_sample_{}_text_{}.wav".format(os.path.splitext(os.path.basename(args.wav_file))[0], texts_name[spec_idx]))
        audio.save_wav(wav, str(wav_fpath), sr=cfg.dataset.sample_rate)


def main():
    parser = argparse.ArgumentParser(description='Streamax SV2TTS Training Engine')

    # # english
    # parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/sv2tts/tts_config_english_sv2tts.py", nargs='?', help='config file')
    # parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/1320_00000.mp3", nargs='?', help='config file')
    # parser.add_argument('-t', '--text', type=str, default="activate be double you see.", nargs='?', help='config file')
    # parser.add_argument('-tn', '--text_name', type=str, default="activate_b_w_c", nargs='?', help='config file')

    # chinese
    # parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/sv2tts/tts_config_chinese_sv2tts.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_1_1_10232021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')
    parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_finetune_1_2_10232021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_tacotron_singlespeaker_guaiding_4_2_10292021/tts_config_chinese_sv2tts.py", nargs='?', help='config file')
    # parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00050001.wav", nargs='?', help='config file')
    parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00730005.wav", nargs='?', help='config file')
    parser.add_argument('-t', '--text', type=str, default=" ".join(get_pinyin("道路千万条安全第一条")), nargs='?', help='config file')
    parser.add_argument('-tn', '--text_name', type=str, default="道路千万条安全第一条", nargs='?', help='config file')
    # parser.add_argument('-t', '--text', type=str, default=" ".join(get_pinyin("今天星期五天气好真开心")), nargs='?', help='config file')
    # parser.add_argument('-tn', '--text_name', type=str, default="今天星期五天气好真开心", nargs='?', help='config file')

    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()