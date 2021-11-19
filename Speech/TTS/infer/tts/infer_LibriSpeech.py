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
    cfg_synthesizer = cfg
    net_synthesizer = import_network(cfg_synthesizer, 
                                    cfg.net.model_name, 
                                    cfg.net.class_name)
    load_checkpoint(net_synthesizer, 
                    0,
                    cfg.general.save_dir, cfg.test.model_epoch, 'checkpoints',
                    "",
                    'state_dict', [], 0)
    net_synthesizer.eval()

    # load text
    texts = args.text_list
    texts_name = args.text_name_list

    # load wav
    speaker_id_list = os.listdir(args.data_path)
    speaker_id_list.sort()
    
    for speaker_idx in tqdm(range(len(speaker_id_list))):
        speaker_id = speaker_id_list[speaker_idx]
        section_id_list = os.listdir(os.path.join(args.data_path, speaker_id))
        section_id_list.sort()

        for section_idx in range(len(section_id_list)):
            section_id = section_id_list[section_idx]
            wav_list = os.listdir(os.path.join(args.data_path, speaker_id, section_id))
            wav_list.sort()

            for time_id in range(args.random_time):
                wav_id = wav_list[random.randint(0, len(wav_list) - 1)]
                if not wav_id.endswith("mp3") and not wav_id.endswith("wav") and not wav_id.endswith("flac"):
                    continue

                wav_path = os.path.join(args.data_path, speaker_id, section_id, wav_id)

                # load embedding
                embed = embedding(cfg_speaker_verification, sv_net, wav_path) 
                # embed = embedding(cfg_speaker_verification, sv_net, args.wav_file) 

                for idx in range(len(texts)):
                    
                    # create_folder 
                    create_folder(os.path.join(args.output_folder, texts_name[idx]))

                    output_wav_path = os.path.join(args.output_folder, texts_name[idx], args.output_format.format(int(speaker_id), int(idx), time_id))
                    if os.path.exists(output_wav_path):
                        continue

                    # synthesize_spectrogram
                    text = texts[idx]
                    spec, align_score = synthesize_spectrogram(cfg_synthesizer, net_synthesizer, text, embed)

                    print("Synthesizing Align Score: " , align_score)
                    if align_score > 0.5:
                        continue

                    ## Generating the waveform
                    print("Synthesizing the waveform: ", text)

                    # save griffin lim inverted wav for debug (mel -> wav)
                    wav = audio.compute_inv_mel_spectrogram(cfg, spec)

                    # Save it on the disk
                    audio.save_wav(wav, str(output_wav_path), sr=cfg.dataset.sample_rate)


def main():
    parser = argparse.ArgumentParser(description='Streamax TTS Training Engine')

    # english
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/tts/tts_config_english_sv2tts.py", nargs='?', help='config file')
    parser.add_argument('-d', "--data_path", type=str, 
                        # default= "/mnt/huanyuan/data/speech/asr/English/LibriSpeech/LibriSpeech/train-clean-100/")
                        default= "/mnt/huanyuan/data/speech/asr/English/LibriSpeech/LibriSpeech/train-clean-360/")
                        # default= "/mnt/huanyuan/data/speech/asr/English/LibriSpeech/LibriSpeech/train-other-500/")
    parser.add_argument("--output_folder", type=str, 
                        # default= "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_BwcKeyword/tts/sv2tts_english_finetune_2_0/LibriSpeech/train-clean-100/")
                        default= "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_BwcKeyword/tts/sv2tts_english_finetune_2_0/LibriSpeech/train-clean-360/")
    parser.add_argument("--output_format", type=str, default= "RM_TTSsv2tts_S{:0>6d}P{:0>3d}T{:0>3d}.wav")
    parser.add_argument("--random_time", type=int, default= 5)
    parser.add_argument('-b', '--vocoder_batched', type=str, default="False", nargs='?', help='config file')

    parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/1320_00000.mp3", nargs='?', help='config file')

    args = parser.parse_args()

    # danbing BwcKeyword
    args.text_list = [
                        "start recording. ",
                        "stop recording. ",
                        "mute audio. ",
                        "unmute audio. ",
                        "shots fired. ",
                        "Freeze. ",
                        "Drop your gun. ",
                        "Drop the gun. ",
                        "Keep your hands on. ",
                        "Put your hands up. ", 
                        "Get down on the ground. "
                        ]
    args.text_name_list = [
                            "start_recording",
                            "stop_recording",
                            "mute_audio",
                            "unmute_audio",
                            "shots_fired",
                            "freeze",
                            "drop_your_gun", 
                            "drop_the_gun",
                            "keep_your_hands_on", 
                            "put_your_hands_up",
                            "get_down_on_the_ground", 
                            ]

    infer(args)


if __name__ == "__main__":
    main()