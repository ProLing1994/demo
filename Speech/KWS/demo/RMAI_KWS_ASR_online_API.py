import numpy as np
import os
import pyaudio
import signal
import wave

from multiprocessing import Process, Event, Queue, freeze_support

from RMAI_KWS_ASR_API import KwsAsrApi


# options 
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_BWC_bpe.py"
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_BWC_phoneme.py"
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_BWC_bpe_phoneme.py"
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_MTA_XIAOAN.py"
cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_MANDARIN_TAXI_4s_16k_64dim.py"
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_MANDARIN_TAXI_4s_8k_56dim.py"
# cfg_path = "/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_cq_taxi_3s.py"

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
    在线音频识别
    """
    audio_queue_play = Queue()
    audio_queue_wakeup = Queue()
    event = Event() 
    
    def __init__(self, chunk=1600, format=pyaudio.paInt16, channels=1, rate=16000):
        self._chunk = chunk
        self._format = format
        self._channels = channels
        self._rate = rate

    def listen(self, event, queue):
        """
        进程：监听录音
        """
        print("[Init:] Listen")
        pyaudio_listen = pyaudio.PyAudio()
        stream = pyaudio_listen.open(format=self._format,
                                    channels=self._channels,
                                    rate=self._rate,
                                    input=True,
                                    frames_per_buffer=self._chunk)
                                     
        while True:
            data = stream.read(self._chunk)
            queue.put(data)
            event.set()

    def listen_file(self, event, queue):
        """
        进程：监听本地音乐
        """
        print("[Init:] Listen")
        wave_path = "/home/huanyuan/code/demo/Speech/API/Kws_weakup_Asr/audio/test-kws-xiaorui-asr-mandarin-taxi_001.wav"
    
        # 打开音频流，output=True 表示音频输出
        pyaudio_play = pyaudio.PyAudio()
        stream = pyaudio_play.open(format=self._format,
                                    channels=self._channels,
                                    rate=self._rate,
                                    output=True,
                                    frames_per_buffer=self._chunk)

        wf = wave.open(wave_path, 'rb')
        while True:
            data = wf.readframes(self._chunk)
            stream.write(data)
            queue.put(data)
            event.set()
            
    def play(self, event, queue):
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
        
        while True:
            if queue.empty():
                event.wait()
            else:
                # play
                data = queue.get()
                stream.write(data)

    def wakeup_asr(self, event, queue):
        """
        进程：语音唤醒 + 关键词检索
        """
        print("[Init:] wakeup & asr")
        
        # init
        kws_asr_api = KwsAsrApi(cfg_path = cfg_path, bool_do_kws_weakup=True, bool_do_asr=True, bool_gpu=True)

        audio_data_list = []
        while True:
            if len(audio_data_list) < kws_asr_api.window_size_samples():
                if queue.empty(): 
                    # print("等待数据中..........")
                    event.wait()
                else:
                    data = queue.get()
                    data_np = np.frombuffer(data, np.int16).astype(np.float32) 
                    
                    if len(audio_data_list) == 0:
                        audio_data_list = data_np.tolist()
                    else:
                        audio_data_list.extend(data_np.tolist())
        
            else:
                kws_asr_api.run_kws_asr(np.array(audio_data_list))
                audio_data_list = []

    def start(self):
        """
        实时多进程语音处理
        """
        signal.signal(signal.SIGTERM, term)
        print("[Information:] Current main-process pid is: {}".format(os.getpid()))
        print("[Information:] If you want to kill the main-process and sub-process, type: kill {}".format(os.getpid()))

        # # 监听
        # listen_process_play = Process(target=self.listen, args=(self.event, self.audio_queue_play))
        # listen_process_play.start()

        # # 播放
        # play_process = Process(target=self.play, args=(self.event, self.audio_queue_play))
        # play_process.start()

        # 监听
        # listen_process_wakeup = Process(target=self.listen, args=(self.event, self.audio_queue_wakeup))
        listen_process_wakeup = Process(target=self.listen_file, args=(self.event, self.audio_queue_wakeup))
        listen_process_wakeup.start()

        # 唤醒
        wakeup_asr_process = Process(target=self.wakeup_asr, args=(self.event, self.audio_queue_wakeup))
        wakeup_asr_process.start()

        # listen_process_play.join()
        # play_process.join()
        listen_process_wakeup.join()
        wakeup_asr_process.join()


if __name__ == '__main__':
    freeze_support()
    online_audio = OnlineAudio()
    online_audio.start()