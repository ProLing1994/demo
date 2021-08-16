import argparse
import os
import sys
import time
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/TTS')
from utils.folder_tools import *

def TTS_pyttsx3(args):
    '''
    pyttsx3
    pip install pyttsx3
    注意，该工具存在 bug，只能手动一步一步生成音频数据
    '''
    import pyttsx3
    
    # init 
    args.rate_list = [20]   # [0, -40, -20, 20]
    args.volume_list = [0]
    args.language_list = ['en-westindies'] # [english, en-scottish, english-north, english_rp, english_wmids, english-us, en-westindies]
    args.output_format = "RM_KWS_ACTIVATEBWC_TTSpyttsx3_S{:0>3d}T{:0>5d}.wav"
    speaker_id = 6
    time_id = 3

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    default_rate = engine.getProperty('rate')
    default_volume = engine.getProperty('volume')
    print("default_rate: ", default_rate, "default_volume: ", default_volume)

    # create_folder 
    output_folder = os.path.join(args.output_dir, 'pyttsx3')
    create_folder(output_folder)

    # 遍历不同的 language
    for voice in tqdm(voices):
        # print(voice)
        # print(voice.id, voice.gender, voice.age)
        # if str(voice.id).startswith("en"):
        if str(voice.id) in args.language_list:
            for rate_idx in args.rate_list:
                for volume_idx in args.volume_list:
                    audio_path = os.path.join(args.output_dir, 'pyttsx3', args.output_format.format(speaker_id, time_id))
                    print(voice.id, default_rate + rate_idx, default_volume + volume_idx)
                    engine.setProperty('voice', voice.id)
                    
                    # unknown bugs
                    engine.setProperty('rate', default_rate + rate_idx)
                    engine.setProperty('volume', default_volume + volume_idx)

                    # there is No Female Voice included with the core pyttsx3 package. 
                    # However you can simulate them by adding '+f1' up to '+f4' on the end of the the voice
                    # engine.setProperty('voice', voice.id + '+f1')
                    
                    # # say
                    # engine.say(args.text)

                    # save
                    engine.save_to_file(args.text, audio_path)
                    time_id += 1
            speaker_id += 1 
            time_id = 0

    # run
    engine.runAndWait()


def TTS_gTTS(args):
    '''
    gTTS
    pip install gTTS
    sudo apt install mpg321
    Google Text to Speech (TTS) API.
    '''
    from gtts import gTTS
    tts = gTTS(text='Good morning', lang='en')
    tts.save("./good.mp3")
    os.system("mpg321 good.mp3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudio Format")
    args = parser.parse_args()
    args.samplerate = 16000
    args.text = "Activate, bwc."
    args.output_dir = "/mnt/huanyuan/data/speech/Recording/RM_Activatebwc/tts"

    # pyttsx3
    # TTS_pyttsx3(args)
    TTS_gTTS(args)