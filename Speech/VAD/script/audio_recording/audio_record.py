import argparse
import glob
import multiprocessing
import numpy as np
import os
import pyaudio
import signal
import time
import wave

from multiprocessing import Process, Event, Queue, freeze_support
from tqdm import tqdm


def term(sig_num, addtion):
    """
    杀死进程，子进程随主进程一起被杀死
    """
    pid = os.getpid()
    pgid = os.getpgid(pid)
    print("[Information:] Current main-process pid is: {}, Group pid is: {}".format(pid, pgid))
    
    # kill group pid
    os.killpg(pgid, signal.SIGKILL)


class OnlineAudio:
    """
    语音播放与录制工具
    """    
    def __init__(self, chunk=1600, format=pyaudio.paInt16, channels=1, rate=16000, record_second_asr=5):
        self._chunk = chunk
        self._format = format
        self._channels = channels
        self._record_channes = args.record_channes
        self._rate = rate
        self._record_second_asr = record_second_asr

        # init 
        self._input_folder = args.input_folder
        self._output_folder = args.output_folder
        self._pause_interval_s = args.pause_interval_s
        self._audio_list = []

        # mkdir 
        if not os.path.exists(self._output_folder):
            os.makedirs(self._output_folder)

        # load audio_list
        print("[Init:] Load audio list: ")
        wave_list = glob.glob(os.path.join(self._input_folder, '*' + args.suffix))
        wave_list += glob.glob(os.path.join(self._input_folder, '*/*' + args.suffix))
        wave_list += glob.glob(os.path.join(self._input_folder, '*/*/*' + args.suffix))
        wave_list += glob.glob(os.path.join(self._input_folder, '*/*/*/*' + args.suffix))
        for idx in tqdm(range(len(wave_list))):
            wave_path = wave_list[idx]

            if not wave_path.endswith(args.suffix):
                continue

            output_path = wave_path.replace(self._input_folder, self._output_folder)
            # tqdm.write("{} -> {}".format(input_path, output_path))

            if os.path.exists(output_path):
                continue

            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            self._audio_list.append(wave_path)
        self._audio_list.sort()

        # event
        # play_ready
        self._play_ready_event = Event() 
        # play：事件用于控制进程 play
        self._play_event = Event() 

        # listen_ready
        self._listen_ready_event = Event() 
        # listen：事件用于控制进程 listen
        self._listen_event = Event() 

        self._play_ready_event.clear()
        self._play_event.clear()
        self._listen_ready_event.clear()
        self._listen_event.clear()


    def play(self):
        """
        进程：播放音乐
        """
        print("[Init:] Play")
        pyaudio_play = pyaudio.PyAudio()
        # 打开音频流， output=True 表示音频输出
        stream = pyaudio_play.open(format=self._format,
                                            channels=self._channels,
                                            rate=self._rate,
                                            output=True,
                                            frames_per_buffer=self._chunk)
        play_idx = 0
        while play_idx < len(self._audio_list):
            wave_path = self._audio_list[play_idx]
            print("[paly: ] {}/{}, {}".format(play_idx, len(self._audio_list), wave_path))

            # 加载语音
            wf = wave.open(wave_path, 'rb')

            # 控制监听与播放同步
            self._play_ready_event.set()
            self._listen_ready_event.wait()

            # 播放语音
            stream.start_stream()
            data = wf.readframes(wf.getnframes())
            stream.write(data)
            stream.stop_stream()

            self._play_ready_event.clear()
            self._play_event.set()
            time.sleep(self._pause_interval_s)
            self._listen_event.wait()
            self._listen_event.clear()
            play_idx += 1
        return


    def listen(self):
        """
        进程：监听录音
        """
        print("[Init:] Listen")
        pyaudio_listen = pyaudio.PyAudio()
        stream = pyaudio_listen.open(format=self._format,
                                    channels=self._record_channes,
                                    rate=self._rate,
                                    input=True,
                                    frames_per_buffer=self._chunk)
        
        frames = []
        listen_idx = 0
        while listen_idx < len(self._audio_list):
            self._listen_ready_event.set()
            self._play_ready_event.wait()

            data = stream.read(self._chunk)
            frames.append(data)

            if(self._play_event.is_set()):
                self._play_event.clear()
                self._listen_ready_event.clear()

                wave_path = self._audio_list[listen_idx]
                wave_path = wave_path.replace(self._input_folder, self._output_folder)
                print("[listen: ] {}/{}, {}".format(listen_idx, len(self._audio_list), wave_path))

                # 保存语音
                wf = wave.open(wave_path, 'wb')
                wf.setnchannels(self._record_channes)
                wf.setsampwidth(pyaudio_listen.get_sample_size(self._format))
                wf.setframerate(self._rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                frames = []
                listen_idx += 1
                self._listen_event.set()
        return



    def start(self):
        """
        实时多进程语音处理
        """
        signal.signal(signal.SIGTERM, term)
        print("[Information:] Current main-process pid is: {}".format(os.getpid()))
        print("[Information:] If you want to kill the main-process and sub-process, type: kill {}".format(os.getpid()))
        
        # 播放
        play_process = Process(target=self.play, args=())
        play_process.start()

        # 监听
        listen_process = Process(target=self.listen, args=())
        listen_process.start()

        play_process.join()
        listen_process.join()
        term


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # freeze_support()

    parser = argparse.ArgumentParser(description='Streamax Engine')
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech_wav/")
    parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech_wav_record/")
    parser.add_argument('--suffix', type=str, default=".wav")
    parser.add_argument('--record_channes', type=int, default=1)
    parser.add_argument('--pause_interval_s', type=float, default=1)
    args = parser.parse_args()

    online_audio = OnlineAudio()
    online_audio.start()