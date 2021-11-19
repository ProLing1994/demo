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

from TTS.utils.tts.infer_tools import *
from TTS.utils.tts.visualizations_tools import *

from TTS.utils.vocoder.infer_tools import *


def infer(args):
    """
    模型推理，通过滑窗的方式得到每一小段 embedding，随后计算 EER
    """
    # load config
    cfg = load_cfg_file(args.config_file)

    # load speaker verification net
    if cfg.dataset.mutil_speaker:
        cfg_speaker_verification = load_cfg_file(cfg.speaker_verification.config_file)
        sv_net = import_network(cfg_speaker_verification, 
                                cfg.speaker_verification.model_name, 
                                cfg.speaker_verification.class_name)
        load_checkpoint(sv_net, 
                        cfg.speaker_verification.load_mode_type,
                        cfg.speaker_verification.model_dir, cfg.speaker_verification.epoch_num, cfg.speaker_verification.sub_folder_name,
                        cfg.speaker_verification.model_path,
                        cfg.speaker_verification.state_name, cfg.speaker_verification.ignore_key_list, cfg.speaker_verification.add_module_type)
        sv_net.eval()

    # load synthesizer net
    cfg_synthesizer = load_cfg_file(cfg.synthesizer.config_file)
    net_synthesizer = import_network(cfg_synthesizer, 
                                    cfg.synthesizer.model_name, 
                                    cfg.synthesizer.class_name)
    load_checkpoint(net_synthesizer, 
                    cfg.synthesizer.load_mode_type,
                    cfg.synthesizer.model_dir, cfg.synthesizer.epoch_num, cfg.synthesizer.sub_folder_name,
                    cfg.synthesizer.model_path,
                    cfg.synthesizer.state_name, cfg.synthesizer.ignore_key_list, cfg.synthesizer.add_module_type)
    net_synthesizer.eval()

    # load vocoder net 
    cfg_vocoder = cfg
    net_vocoder = import_network(cfg_vocoder, 
                                    cfg.net.model_name, 
                                    cfg.net.class_name)
    load_checkpoint(net_vocoder, 
                    0,
                    cfg.general.save_dir, cfg.test.model_epoch, 'checkpoints',
                    "",
                    'state_dict', [], 0)
    net_vocoder.eval()

    # load text
    texts = args.text_list
    texts_name = args.text_name_list

    # load embedding
    if cfg.dataset.mutil_speaker:
        embed = embedding(cfg_speaker_verification, sv_net, args.wav_file) 
    else:
        embed = None


    for idx in tqdm(range(len(texts))):
        # synthesize_spectrogram
        text = texts[idx]
        spec, align_score = synthesize_spectrogram(cfg_synthesizer, net_synthesizer, text, embed)

        ## Generating the waveform
        print("Synthesizing Align Score: " , align_score)
        print("Synthesizing the waveform: ", text)

        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        if args.vocoder_batched == 'True':
            generated_wav = infer_waveform(net_vocoder, spec)
        else:
            generated_wav = infer_waveform(net_vocoder, spec, batched=False)
        
        ## Post-generation
        # Trim excess silences to compensate for gaps in spectrograms
        generated_wav = audio.preprocess_wav(generated_wav, cfg.dataset.sample_rate)

        # Save it on the disk
        output_dir = os.path.join(cfg.general.save_dir, "infer")
        create_folder(output_dir)
        wav_fpath = os.path.join(output_dir, "wave_wavernn_from_mel_sample_{}_text_{}_batched_{}.wav".format(os.path.splitext(os.path.basename(args.wav_file))[0], texts_name[idx], str(args.vocoder_batched)))
        audio.save_wav(generated_wav, str(wav_fpath), sr=cfg.dataset.sample_rate)


def main():
    parser = argparse.ArgumentParser(description='Streamax TTS Training Engine')

    # english
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/vocoder/tts_config_english_vocoder_wavernn.py", nargs='?', help='config file')
    parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/1320_00000.mp3", nargs='?', help='config file')
    parser.add_argument('-b', '--vocoder_batched', type=str, default="False", nargs='?', help='config file')
    # parser.add_argument('-b', '--vocoder_batched', type=str, default="True", nargs='?', help='config file')
    args = parser.parse_args()
    args.text_list = [
                        "activate be double you see.", 
                        "start recording start recording start recording start recording start recording. ",
                        "stop recording stop recording stop recording stop recording stop recording. ",
                        "mute audio mute audio mute audio mute audio mute audio. ",
                        "unmute audio unmute audio unmute audio unmute audio unmute audio. ",
                        ]
    args.text_name_list = [
                            "activate_b_w_c", 
                            "start_recording",
                            "stop_recording",
                            "mute_audio",
                            "unmute_audio",
                            ]

    # # chinese
    # # parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/vocoder/tts_config_chinese_vocoder_wavernn.py", nargs='?', help='config file')
    # # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts_vocoder/chinese_tts_vocoder/wavernn_chinese_mutil_speaker_1_0_11012021/tts_config_chinese_mutil_speaker_vocoder_wavernn.py", nargs='?', help='config file')
    # # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts_vocoder/chinese_tts_vocoder/wavernn_chinese_mutil_speaker_1_0_11012021/tts_config_chinese_signal_speaker_vocoder_wavernn.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--config_file', type=str, default="/mnt/huanyuan2/model/tts_vocoder/chinese_tts_vocoder/wavernn_chinese_single_speaker_1_0_11032021/tts_config_chinese_signal_speaker_vocoder_wavernn.py", nargs='?', help='config file')
    # parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00050001.wav", nargs='?', help='config file')
    # # parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/Aishell3/SSB00730005.wav", nargs='?', help='config file')
    # # parser.add_argument('-b', '--vocoder_batched', type=str, default="False", nargs='?', help='config file')
    # parser.add_argument('-b', '--vocoder_batched', type=str, default="True", nargs='?', help='config file')
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

    infer(args)


if __name__ == "__main__":
    main()