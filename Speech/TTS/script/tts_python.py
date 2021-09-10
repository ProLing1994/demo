import argparse
from enum import Flag
import os
import sys
import time
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo/Speech/TTS')
sys.path.insert(0, r"E:\project\demo\Speech\TTS")
from utils.folder_tools import *

'''
tts python api
https://pythonprogramminglanguage.com/text-to-speech/
'''

def TTS_pyttsx3(args):
    '''
    pyttsx3
    pip install pyttsx3
    注意，该工具存在 bug，只能手动一步一步生成音频数据
    缺点：不包含女生音频，音质较差
    https://blog.csdn.net/u014663232/article/details/103834543
    '''
    import pyttsx3
    # demo
    # engine = pyttsx3.init()
    # engine.say('Good morning.')
    # engine.runAndWait()

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
    Google Text to Speech (TTS) API.
    '''
    from gtts import gTTS
    # demo
    # tts = gTTS(text=args.text, lang='en')
    # tts = gTTS(text=args.text, lang='en', slow=True)
    # tts.save("./good.mp3")

    # init 
    args.slow_list = [True, False]
    args.language_list = ['en'] # [en]
    args.language_tld_list = ['com.au', 'co.uk', 'com', 'ca', 'co.in', 'ie', 'co.za']
    args.output_format = "RM_KWS_ACTIVATEBWC_TTSgTTS_S{:0>3d}T{:0>5d}.wav"
    speaker_id = 0
    time_id = 0

    # create_folder 
    output_folder = os.path.join(args.output_dir, 'gTTS')
    create_folder(output_folder)

    for language_idx in args.language_list:
        for language_tld_idx in args.language_tld_list:
            for slow_idx in args.slow_list:
                audio_path = os.path.join(args.output_dir, 'gTTS', args.output_format.format(speaker_id, time_id))
                print(language_idx, language_tld_idx, slow_idx)

                tts = gTTS(text=args.text, lang=language_idx, tld=language_tld_idx, slow=slow_idx)
                tts.save(audio_path)
                
                time_id += 1

            speaker_id += 1 
            time_id = 0


def TTS_win32com(args):
    '''
    win32com
    Microsoft Windows 10, Microsoft speech engine.
    '''
    import win32com.client as wincl
    speak = wincl.Dispatch("SAPI.SpVoice")
    speak.Speak(args.text)


def TTS_watson(args):
    '''
    tts-watson
    pip install --upgrade "ibm-watson>=5.2.2"
    https://cloud.ibm.com/apidocs/text-to-speech?code=python
    IBM tts API.
    '''
    import json
    from ibm_watson import TextToSpeechV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

    # demo
    # authenticator = IAMAuthenticator('hCGxBbsosLhiraIpCRQCq70hirhWk_Cy2K42_t2hoTYM')
    # text_to_speech = TextToSpeechV1(
    #     authenticator=authenticator
    # )
    # text_to_speech.set_service_url('https://api.au-syd.text-to-speech.watson.cloud.ibm.com/instances/70f0fead-74bc-4331-911c-524b954c27ac')
    # voices = text_to_speech.list_voices().get_result()
    # print(json.dumps(voices, indent=2))
    # with open('./hello_world.wav', 'wb') as audio_file:
    #     audio_file.write(
    #         text_to_speech.synthesize(
    #             args.text,
    #             voice='en-US_AllisonV3Voice',
    #             accept='audio/wav'        
    #         ).get_result().content)

    # init 
    args.language_list = ['en-AU_CraigVoice', 'en-AU_MadisonVoice', 'en-GB_CharlotteV3Voice', 'en-GB_JamesV3Voice', 'en-GB_KateV3Voice',
                            'en-US_AllisonV3Voice', 'en-US_EmilyV3Voice', 'en-US_HenryV3Voice', 'en-US_KevinV3Voice', 'en-US_LisaV3Voice',
                            'en-US_MichaelV3Voice', 'en-US_OliviaV3Voice']
    args.output_format = "RM_KWS_ACTIVATEBWC_TTSwatson_S{:0>3d}T{:0>5d}.wav"
    speaker_id = 0
    time_id = 0
    
    # create_folder 
    output_folder = os.path.join(args.output_dir, 'watson')
    create_folder(output_folder)

    authenticator = IAMAuthenticator('hCGxBbsosLhiraIpCRQCq70hirhWk_Cy2K42_t2hoTYM')
    text_to_speech = TextToSpeechV1(
        authenticator=authenticator
    )
    text_to_speech.set_service_url('https://api.au-syd.text-to-speech.watson.cloud.ibm.com/instances/70f0fead-74bc-4331-911c-524b954c27ac')

    for language_idx in args.language_list:
        audio_path = os.path.join(args.output_dir, 'watson', args.output_format.format(speaker_id, time_id))
        print(language_idx)

        with open(audio_path, 'wb') as audio_file:
            audio_file.write(
                text_to_speech.synthesize(
                    args.text,
                    voice=language_idx,
                    accept='audio/wav'        
                ).get_result().content)
        speaker_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudio Format")
    args = parser.parse_args()
    args.samplerate = 16000
    args.text = "Activate, bwc."
    # args.output_dir = "/mnt/huanyuan/data/speech/Recording/RM_Activatebwc/tts"
    args.output_dir = r"\\192.168.151.112\mnt\data\speech\Recording\RM_Activatebwc\tts"

    # TTS_pyttsx3(args)
    # TTS_gTTS(args)
    # TTS_win32com(args)
    TTS_watson(args)