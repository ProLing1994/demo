import matplotlib.pyplot as plt
import numpy as np
import os
import pyaudio
import signal
import sys
import wave

from matplotlib import animation
from multiprocessing import Process, Event, Queue, freeze_support

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *
from dataset.kws.dataset_helper import *
from impl.pred_pyimpl import kws_load_model, model_predict
from impl.recognizer_pyimpl import RecognizeResult, RecognizeCommands


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


    def display(self, event, queue):
        """
        进程：绘制图像
        """
        print("[Init:] display")
        cnt = 0

        while True:
            if queue.empty():
                # print("等待数据中..........")
                event.wait()
            else:
                # play
                data = queue.get()
                # print("获取的数据data", data)
                data_np = np.fromstring(data, dtype = np.int16)

                # First set up the figure, the axis, and the plot element we want to animate
                fig = plt.figure()
                ax = plt.axes(xlim=(0, len(data_np)), ylim=(-50000, 50000))
                line, = ax.plot([], [], lw=2)
                 
                # initialization function: plot the background of each frame
                def init():
                    line.set_data([], [])
                    return line,
                 
                # animation function.  This is called sequentially
                # note: num is framenumber
                def animate(num):
                    data = queue.get()
                    data_np = np.fromstring(data, dtype=np.int16)
                    print(data_np.shape)
                    x = range(len(data_np))
                    y = data_np
                    line.set_data(x, y)
                    return line,
                 
                # call the animator.  blit=True means only re-draw the parts that have changed.
                anim = animation.FuncAnimation(fig, animate, init_func=init, blit=True)                 
                # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264']) 

                if cnt == 0:
                    plt.show()
                else:
                    continue
                cnt += 1
 

    def fit(self, event, queue):
        print('[Init:] fit')
        while True:
            if queue.empty():
                # print("等待数据中..........")
                event.wait()
            else:
                data = queue.get()
                # print("获取的数据data", data)
                data_np = np.fromstring(data,dtype = np.int16)

                max_freq = np.max(data_np)#给PHP
                # print([np.max(data1),np.std(data1)])
                if np.max(data_np) > 8000 :
                    print ("检测到异常信号")
                    print ('当前信号：', max_freq)
                    #存入音频文件
                    clc = int(time.time())
                    file = wave.open(str(clc)+".wav", "wb")
                    file.setnchannels(1)
                    file.setframerate(2000)
                    file.setnframes(10000)
                    file.setsampwidth(2)
                    file.writeframes(data_np)
                    file.close()                
                    
                else:
                    print("信号正常")


    def wake_up(self, event, queue):
        print('[Init:] wake up')

        # config
        config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu5_1_fbank_timeshift_spec_on_res15_11032020/test_straming_wav/kws_config_xiaoyu_2.py"
        # config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu3_3_timeshift_spec_on_focal_res15_11032020/test_straming_wav/kws_config_xiaoyu_2.py"
        cfg = load_cfg_file(config_file)
        label_index = load_label_index(cfg.dataset.label.positive_label)
        label_list = cfg.dataset.label.label_list
        positive_label = cfg.dataset.label.positive_label
        sample_rate = cfg.dataset.sample_rate
        clip_duration_ms = cfg.dataset.clip_duration_ms

        # inin model parameter
        model_path = cfg.general.save_dir
        gpu_ids = int(cfg.general.gpu_ids)
        model_epoch = -1
        
        # init parameter 
        # detection_threshold = 0.95
        detection_threshold = 0.5
        timeshift_ms = 30
        average_window_duration_ms = 800
        audio_data_length = 0
        audio_data_offset = 0

        desired_samples = int(sample_rate * clip_duration_ms / 1000)
        timeshift_samples = int(sample_rate * timeshift_ms / 1000)

        # init recognizer
        recognize_element = RecognizeResult()
        recognize_commands = RecognizeCommands(
            labels=label_list,
            positove_lable_index = label_index[positive_label[0]],
            average_window_duration_ms=average_window_duration_ms,
            detection_threshold=detection_threshold,
            suppression_ms=3000,
            minimum_count=15)

        # init model
        model = kws_load_model(model_path, gpu_ids, model_epoch)
        net = model['prediction']['net']
        net.eval()

        while True:
            if audio_data_length < desired_samples:
                if queue.empty(): 
                    # print("等待数据中..........")
                    event.wait()
                else:
                    data = queue.get()
                    # 数据转化，注意除以 2^15 进行归一化
                    data_np = np.frombuffer(data, np.int16).astype(np.float32) / 32768
                    
                    if audio_data_length == 0:
                        audio_data = data_np
                        audio_data_length = len(audio_data)
                    else:
                        audio_data = np.concatenate((audio_data, data_np), axis=0)
                        audio_data_length = len(audio_data)
            else:
                # prepare data
                input_data = audio_data[0: desired_samples]
                assert len(input_data) == desired_samples

                # model infer
                output_score = model_predict(cfg, net, input_data)

                # process result
                current_time_ms = int(audio_data_offset * 1000 / sample_rate)
                recognize_commands.process_latest_result(output_score, current_time_ms, recognize_element)

                # print(output_score[0])
                if recognize_element.is_new_command:
                    all_found_words_dict = {}
                    all_found_words_dict['label'] = positive_label[0]
                    all_found_words_dict['start_time'] = recognize_element.start_time
                    all_found_words_dict['end_time'] = recognize_element.end_time + clip_duration_ms
                    print('Find words: label:{}, start time:{}, end time:{}, response time: {:.2f}s'.format(
                        all_found_words_dict['label'], all_found_words_dict['start_time'], all_found_words_dict['end_time'], recognize_element.response_time))

                audio_data_offset += timeshift_samples
                audio_data = audio_data[timeshift_samples:]
                audio_data_length = len(audio_data)


    def start(self):
        """
        实时多进程语音处理
        """
        signal.signal(signal.SIGTERM, term)
        print("[Information:] Current main-process pid is: {}".format(os.getpid()))
        print("[Information:] If you want to kill the main process and sub-process, type: kill {}".format(os.getpid()))

        # # 监听
        # listen_process_play = Process(target=self.listen, args=(self.event, self.audio_queue_play))
        # listen_process_play.start()

        # # 播放
        # play_process = Process(target=self.play, args=(self.event, self.audio_queue_play))
        # play_process.start()

        # 监听
        listen_process_wakeeup = Process(target=self.listen, args=(self.event, self.audio_queue_wakeup))
        listen_process_wakeeup.start()

        # 唤醒
        wakeup_process = Process(target=self.wake_up, args=(self.event, self.audio_queue_wakeup))
        wakeup_process.start()

        # # 绘图
        # display_process = Process(target=self.display, args=(self.event, self.audio_queue_wakeup))
        # display_process.start()

        # # Judge and alarm
        # judge_alarm_process = Process(target=self.fit,args=(self.event, self.audio_queue_play))
        # judge_alarm_process.start()

        # listen_process_play.join()
        # play_process.join()
        listen_process_wakeeup.join()
        wakeup_process.join()
        # display_process.join()
        # judge_alarm_process.join()


if __name__ == '__main__':
    freeze_support()
    online_audio = OnlineAudio()
    online_audio.start()

