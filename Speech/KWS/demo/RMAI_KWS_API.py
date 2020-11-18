import numpy as np
import os
import pyaudio
import signal
import wave

from multiprocessing import Process, Event, Queue, freeze_support

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
        # wave_path = "/mnt/huanyuan/model/test_straming_wav/xiaoyu_03022018_testing_60_001.wav"
        wave_path = "/mnt/huanyuan/model/test_straming_wav/test.wav"
    
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

    def wake_up(self, event, queue):
        """
        进程：语音唤醒
        """
        print("[Init:] wake up")
        wake_up = WakeUp()

        while True:
            if wake_up.audio_data_length < wake_up.desired_samples:
                if queue.empty(): 
                    # print("等待数据中..........")
                    event.wait()
                else:
                    data = queue.get()
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
                    print("[Information:] Waked up! Wake up word: {}, Time: {}".format(wake_up.positive_label_list[0], wake_up.audio_data_offset/wake_up.sample_rate))

                wake_up.audio_data_offset += wake_up.timeshift_samples
                wake_up.audio_data = wake_up.audio_data[wake_up.timeshift_samples:]
                wake_up.audio_data_length = len(wake_up.audio_data)

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
        listen_process_wakeeup = Process(target=self.listen, args=(self.event, self.audio_queue_wakeup))
        # listen_process_wakeeup = Process(target=self.listen_file, args=(self.event, self.audio_queue_wakeup))
        listen_process_wakeeup.start()

        # 唤醒
        wakeup_process = Process(target=self.wake_up, args=(self.event, self.audio_queue_wakeup))
        wakeup_process.start()

        # listen_process_play.join()
        # play_process.join()
        listen_process_wakeeup.join()
        wakeup_process.join()


if __name__ == '__main__':
    freeze_support()
    online_audio = OnlineAudio()
    online_audio.start()