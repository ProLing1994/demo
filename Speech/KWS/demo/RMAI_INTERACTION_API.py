import numpy as np
import os
import pyaudio
import signal
import wave

from multiprocessing import Process, Event, Queue, freeze_support
from tqdm import tqdm

from wakeup.wake_up import WakeUp


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
    audio_queue = Queue()
    audio_event = Event()

    # play：事件用于控制进程 play
    play_event = Event()       
    # wake_up：事件用于控制进程 wake_up
    wake_up_event = Event()
    
    def __init__(self, chunk=1600, format=pyaudio.paInt16, channels=1, rate=16000, record_second_asr=5):
        self._chunk = chunk
        self._format = format
        self._channels = channels
        self._rate = rate
        self._record_second_asr = record_second_asr

        # init
        self.wake_up_event.set()

    def listen(self, audio_queue, audio_event):
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
            audio_queue.put(data)
            audio_event.set()
            
    def play(self, audio_queue, audio_event, play_event, wake_up_event):
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
        # asr 语音录制时间, 5s
        record_times = int(self._rate / self._chunk * self._record_second_asr)
        
        while True:
            play_event.wait()

            # 语音录制
            print("[Information:] 语音录制：")
            audio_data_list = []
            for _ in tqdm(range(record_times)):
                data = audio_queue.get()
                audio_data_list.append(data)
            print("[Information:] 语音录制：Done")
            
            # 语音播放
            print("[Information:] 语音播放：")
            for idx in tqdm(range(record_times)):
                data = audio_data_list[idx]
                stream.write(data)
            print("[Information:] 语音播放：Done, Play time: {:.2f}s".format(audio_queue.qsize() * 1.0 * self._chunk / self._rate))

            # 清空当前操作过程中，录音进程所录制的声音
            audio_queue_size = audio_queue.qsize()
            for _ in range(max(0, audio_queue_size - int((self._rate / self._chunk * 2.0)))):      # -2s, soft, 否则唤醒任务启动时间长(目前语音唤醒对 2s 音频建模)
                audio_queue.get()

            play_event.clear()
            wake_up_event.set()

    def wake_up(self, audio_queue, audio_event, play_event, wake_up_event):
        """
        进程：语音唤醒
        """
        print("[Init:] Wake up")
        wake_up = WakeUp(config_file=config_file)

        while True:
            wake_up_event.wait()

            if wake_up.audio_data_length < wake_up.desired_samples:
                if audio_queue.empty(): 
                    audio_event.wait()
                else:
                    data = audio_queue.get()
                    # 数据转化，注意除以 2^15 进行归一化
                    data_np = np.frombuffer(data, np.int16).astype(np.float32) / 32768
                    
                    if wake_up.audio_data_length == 0:
                        wake_up.audio_data = data_np
                        wake_up.audio_data_length = len(wake_up.audio_data)
                    else:
                        wake_up.audio_data = np.concatenate((wake_up.audio_data, data_np), axis=0)
                        wake_up.audio_data_length = len(wake_up.audio_data)
            else:
                # prepare data
                input_data = wake_up.audio_data[0: wake_up.desired_samples]
                assert len(input_data) == wake_up.desired_samples

                # model infer
                find_word_bool = wake_up.predict(input_data)

                if find_word_bool:
                    print("[Information:] Waked up! Wake up word: {}, Time: {}, Wake up time: {:.2f}s".format(wake_up.positive_label_list[0], 
                                                                                                                wake_up.audio_data_offset/wake_up.sample_rate, 
                                                                                                                audio_queue.qsize() * 1.0 * self._chunk / self._rate))
                    wake_up.audio_data = None
                    wake_up.audio_data_length = 0

                    # 清空当前操作过程中，录音进程所录制的声音
                    audio_queue_size = audio_queue.qsize()
                    for _ in range(audio_queue_size):
                        audio_queue.get()

                    wake_up_event.clear()
                    play_event.set()
                else:
                    wake_up.audio_data = wake_up.audio_data[wake_up.timeshift_samples:]
                    wake_up.audio_data_length = len(wake_up.audio_data)

                wake_up.audio_data_offset += wake_up.timeshift_samples

    def start(self):
        """
        实时多进程语音处理
        """
        signal.signal(signal.SIGTERM, term)
        print("[Information:] Current main-process pid is: {}".format(os.getpid()))
        print("[Information:] If you want to kill the main-process and sub-process, type: kill {}".format(os.getpid()))

        # 监听
        listen_process_wakeup = Process(target=self.listen, args=(self.audio_queue, self.audio_event))
        listen_process_wakeup.start()

        # 唤醒
        wakeup_process = Process(target=self.wake_up, args=(self.audio_queue, self.audio_event, self.play_event, self.wake_up_event))
        wakeup_process.start()

        # 播放
        play_process = Process(target=self.play, args=(self.audio_queue, self.audio_event, self.play_event, self.wake_up_event))
        play_process.start()

        listen_process_wakeup.join()
        wakeup_process.join()
        play_process.join()


if __name__ == '__main__':
    # config_file="./wakeup/wake_up_xiaoyu_config.py"
    config_file="./wakeup/wake_up_xiaorui_config.py"

    freeze_support()
    online_audio = OnlineAudio()
    online_audio.start()