import argparse
import librosa
import numpy as np
import os
from pathlib import Path
import random
import soundfile as sf
import sys
from tqdm import tqdm
import torch 

SV2TTS_path = '/home/huanyuan/code/third_code/Real-Time-Voice-Cloning/'
sys.path.insert(0, SV2TTS_path)
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from TTS.utils.folder_tools import *

def embedding(in_fpath):
    ## Computing the embedding
    # First, we load the wav using the function that the speaker encoder provides. This is 
    # important: there is preprocessing that must be applied.
    
    # The following two methods are equivalent:
    # - Directly load from the filepath:
    preprocessed_wav = encoder.preprocess_wav(in_fpath)
    # - If the wav is already loaded:
    original_wav, sampling_rate = librosa.load(str(in_fpath))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    # print("Loaded file succesfully")
    
    # Then we derive the embedding. There are many functions and parameters that the 
    # speaker encoder interfaces. These are mostly for in-depth research. You will typically
    # only use this function (with its default parameters):
    embed = encoder.embed_utterance(preprocessed_wav)
    # print("Created the embedding")
    return embed

def speech_generation_LibriSpeech(args):
    # create_folder 
    create_folder(args.output_folder)

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)
    
    speaker_id_list = os.listdir(args.LibriSpeech_path)
    speaker_id_list.sort()
    for speaker_idx in tqdm(range(len(speaker_id_list))):
        speaker_id = speaker_id_list[speaker_idx]
        section_id_list = os.listdir(os.path.join(args.LibriSpeech_path, speaker_id))
        section_id_list.sort()
        for section_idx in range(len(section_id_list)):
            section_id = section_id_list[section_idx]
            wav_list = os.listdir(os.path.join(args.LibriSpeech_path, speaker_id, section_id))
            wav_list.sort()

            # init 
            time_id = 0
            embed_list = []
            
            for wav_idx in range(len(wav_list)):
                wav_id = wav_list[wav_idx]
                if not wav_id.endswith("mp3") and not wav_id.endswith("wav") and not wav_id.endswith("flac"):
                    continue
                wav_path = os.path.join(args.LibriSpeech_path, speaker_id, section_id, wav_id)
                tqdm.write("speaker_id: {}, section_id: {}, wav_id: {}".format(speaker_id, section_id, wav_id))
                embed = embedding(wav_path) 
                embed_list.append(embed)

            # Compute the utterance embedding from multi audio
            raw_embed = np.mean(np.array(embed_list), axis=0)
            embed = raw_embed / np.linalg.norm(raw_embed, 2)

            for random_seed_idx in range(args.random_time):
                random_seed = random_seed_idx

                # If seed is specified, reset torch seed and force synthesizer reload
                torch.manual_seed(random_seed)
                synthesizer = Synthesizer(args.syn_model_fpath)
                vocoder.load_model(args.voc_model_fpath)

                # The synthesizer works in batch, so you need to put your data in a list or numpy array
                texts = [args.text_list[text_idx] for text_idx in range(len(args.text_list))]
                embeds = [embed] * len(args.text_list)
                # If you know what the attention layer alignments are, you can retrieve them here by
                # passing return_alignments=True
                specs = synthesizer.synthesize_spectrograms(texts, embeds)
                print("Created the mel spectrogram")

                for spec_idx in range(len(specs)):
                    spec = specs[spec_idx]

                    ## Generating the waveform
                    print("Synthesizing the waveform:")

                    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
                    # spectrogram, the more time-efficient the vocoder.
                    generated_wav = vocoder.infer_waveform(spec)
                    
                    
                    ## Post-generation
                    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
                    # pad it.
                    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

                    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
                    generated_wav = encoder.preprocess_wav(generated_wav)
                        
                    # Save it on the disk
                    audio_path = os.path.join(args.output_folder, args.output_format.format(int(speaker_id), int(section_id), time_id))
                    print(generated_wav.dtype)
                    sf.write(audio_path, generated_wav.astype(np.float32), synthesizer.sample_rate)
                    time_id += 1
                    print("\nSaved output as %s\n\n" % audio_path)
            
            # output txt
            with open(os.path.join(args.output_folder, "sv2tts.txt"), "a") as f :
                f.write("speaker: {}, equipment_id: {}. \n".format(speaker_id, speaker_idx, section_id))


def speech_generation_RM_Meiguo_BwcKeyword(args):
    # create_folder 
    create_folder(args.output_folder)

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)
    
    speaker_id_list = os.listdir(args.data_path)
    speaker_id_list.sort()
    for speaker_idx in tqdm(range(len(speaker_id_list))):
        speaker_id = speaker_id_list[speaker_idx]
        equipment_id_list = os.listdir(os.path.join(args.data_path, speaker_id))
        equipment_id_list.sort()
        for equipment_idx in range(len(equipment_id_list)):
            equipment_id = equipment_id_list[equipment_idx]
            wav_list = os.listdir(os.path.join(args.data_path, speaker_id, equipment_id))
            wav_list.sort()

            # init 
            time_id = 0

            for _ in range(args.random_time):

                wav_id = wav_list[random.randint(0, len(wav_list) - 1)]
                if not wav_id.endswith("mp3") and not wav_id.endswith("wav") and not wav_id.endswith("flac"):
                    continue
                wav_path = os.path.join(args.data_path, speaker_id, equipment_id, wav_id)
                print("speaker_id: {}, section_id: {}, wav_id: {}".format(speaker_id, equipment_id, wav_id))
                embed = embedding(wav_path) 

                # The synthesizer works in batch, so you need to put your data in a list or numpy array
                texts = [args.text_list[text_idx] for text_idx in range(len(args.text_list))]
                embeds = [embed] * len(args.text_list)
                # If you know what the attention layer alignments are, you can retrieve them here by
                # passing return_alignments=True
                specs = synthesizer.synthesize_spectrograms(texts, embeds)
                print("Created the mel spectrogram")

                for spec_idx in range(len(specs)):
                    spec = specs[spec_idx]

                    ## Generating the waveform
                    print("Synthesizing the waveform:")

                    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
                    # spectrogram, the more time-efficient the vocoder.
                    generated_wav = vocoder.infer_waveform(spec, batched=False)
                    
                    ## Post-generation
                    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
                    # pad it.
                    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

                    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
                    generated_wav = encoder.preprocess_wav(generated_wav)
                        
                    # Save it on the disk
                    audio_path = os.path.join(args.output_folder, args.output_format.format(int(speaker_idx), int(equipment_idx), time_id))
                    print(generated_wav.dtype)
                    sf.write(audio_path, generated_wav.astype(np.float32), synthesizer.sample_rate)
                    time_id += 1
                    print("\nSaved output as %s\n\n" % audio_path)

            # output txt
            with open(os.path.join(args.output_folder, "sv2tts.txt"), "a") as f :
                f.write("speaker: {}, speaker_idx: {}, equipment_id: {}, equipment_idx: {}. \n".format(speaker_id, speaker_idx, equipment_id, equipment_idx))


if __name__ == '__main__':
    """
    多说话人语音合成系统 Multispeaker Text-To-Speech Synthesis
    使用多说话人语音合成系统合成唤醒词语音 activatebwc，用于实现数据增强
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default= SV2TTS_path + "encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path, 
                        default= SV2TTS_path + "synthesizer/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                        default= SV2TTS_path + "vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    
    # # LibriSpeech, example
    # parser.add_argument("--LibriSpeech_path", type=Path, 
    #                     # default= "/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/train-clean-100/")
    #                     # default= "/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/train-clean-360/")
    #                     default= "/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/train-other-500/")
    #                     # default= "/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/test-clean/")
    # parser.add_argument("--output_folder", type=Path, 
    #                     # default= "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/LibriSpeech/train-clean-100/")
    #                     # default= "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/LibriSpeech/train-clean-360/")
    #                     default= "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/LibriSpeech/train-other-500/")
    #                     # default= "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/LibriSpeech/test-clean/")
    # parser.add_argument("--output_format", type=str, 
    #                     default= "RM_KWS_ACTIVATEBWC_TTSsv2tts_S{:0>6d}P{:0>8d}T{:0>6d}.wav")
    # parser.add_argument("--text_list", type=list, 
    #                     default= ["activate be double you see."])
    # parser.add_argument("--random_time", type=int, 
    #                     default= 3)
    # args = parser.parse_args()
    # print_args(args, parser)

    # ## speech generation
    # speech_generation_LibriSpeech(args)


    # RM_Meiguo_BwcKeyword, 深圳录音, example
    parser.add_argument("--data_path", type=Path, 
                        default= "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_BwcKeyword/office/danbing_16k/all/")
    parser.add_argument("--output_folder", type=Path, 
                        default= "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/RM_Meiguo_BwcKeyword/danbing_16k/all/")
    parser.add_argument("--output_format", type=str, 
                        default= "RM_KWS_ACTIVATEBWC_TTSsv2tts_S{:0>6d}P{:0>8d}T{:0>6d}.wav")
    parser.add_argument("--text_list", type=list, 
                        default= ["activate be double you see."])
    parser.add_argument("--random_time", type=int, 
                        default= 10)
    args = parser.parse_args()
    print_args(args, parser)

    ## speech generation
    speech_generation_RM_Meiguo_BwcKeyword(args)