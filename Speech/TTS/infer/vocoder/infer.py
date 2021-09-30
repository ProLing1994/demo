import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *

from SV.utils.infer_tools import *

from TTS.utils.sv2tts.infer_tools import *
from TTS.utils.sv2tts.visualizations_tools import *

from TTS.utils.vocoder.infer_tools import *


def infer(args):
    """
    模型推理，通过滑窗的方式得到每一小段 embedding，随后计算 EER
    """
    # load config
    cfg = load_cfg_file(args.config_file)

    # load speaker verification net
    cfg_speaker_verification = load_cfg_file(cfg.speaker_verification.config_file)
    sv_net = import_network(cfg_speaker_verification, 
                            cfg.speaker_verification.model_name, 
                            cfg.speaker_verification.class_name)
    load_checkpoint(sv_net, 
                    cfg.speaker_verification.epoch, 
                    cfg.speaker_verification.model_dir)
    sv_net.eval()

    # load synthesizer net
    cfg_synthesizer = load_cfg_file(cfg.synthesizer.config_file)
    net_synthesizer = import_network(cfg_synthesizer, 
                                    cfg.synthesizer.model_name, 
                                    cfg.synthesizer.class_name)
    load_checkpoint(net_synthesizer, 
                    cfg.synthesizer.epoch, 
                    cfg.synthesizer.model_dir)
    net_synthesizer.eval()

    # load vocoder net 
    cfg_vocoder = cfg
    net_vocoder = import_network(cfg_vocoder, 
                                    cfg.net.model_name, 
                                    cfg.net.class_name)
    load_checkpoint(net_vocoder, 
                    cfg.test.model_epoch, 
                    cfg.general.save_dir)
    net_vocoder.eval()

    # load embedding
    embed = embedding(cfg_speaker_verification, sv_net, args.wav_file) 
    embeds = [embed]

    # load text
    texts = [args.text]

    # synthesize_spectrograms
    specs = synthesize_spectrograms(cfg_synthesizer, net_synthesizer, texts, embeds)
    
    for spec_idx in range(len(specs)):

        spec = specs[spec_idx]

        ## Generating the waveform
        print("Synthesizing the waveform: ", texts[spec_idx])

        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        if args.vocoder_batched == 'True':
            generated_wav = infer_waveform(net_vocoder, spec)
        else:
            generated_wav = infer_waveform(net_vocoder, spec, batched=False)
        
        ## Post-generation
        # Trim excess silences to compensate for gaps in spectrograms
        generated_wav = preprocess_wav(generated_wav, cfg.dataset.sample_rate)

        # Save it on the disk
        output_dir = os.path.join(cfg.general.save_dir, "infer")
        create_folder(output_dir)
        wav_fpath = os.path.join(output_dir, "wave_from_mel_sample_{}_text_{}_batched_{}.wav".format(os.path.splitext(os.path.basename(args.wav_file))[0], texts[spec_idx], str(args.vocoder_batched)))
        save_wav(generated_wav, str(wav_fpath), sr=cfg.dataset.sample_rate)


def main():
    parser = argparse.ArgumentParser(description='Streamax SV2TTS Training Engine')
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/vocoder/tts_config_vocoder_wavernn.py", nargs='?', help='config file')
    parser.add_argument('-w', '--wav_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/infer/sample/1320_00000.mp3", nargs='?', help='config file')
    parser.add_argument('-t', '--text', type=str, default="activate be double you see.", nargs='?', help='config file')
    # parser.add_argument('-b', '--vocoder_batched', type=str, default="False", nargs='?', help='config file')
    parser.add_argument('-b', '--vocoder_batched', type=str, default="True", nargs='?', help='config file')
    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()