import argparse
from datetime import datetime
import pyaudio
import wave
import os

def listen(args):
    """
    进程：监听录音
    """
    print("[Init:] Listen")
    pyaudio_listen = pyaudio.PyAudio()
    pyaudio_listen.get_device_info_by_index(1) 
    stream = pyaudio_listen.open(format=pyaudio.paInt16,
                                channels=args.channes,
                                rate=args.sample_rate,
                                input=True,
                                input_device_index=1,
                                frames_per_buffer=args.chunk)
    
    frames = []
    while True:
        data = stream.read(args.chunk)
        frames.append(data)

        if(len(frames) > args.sample_rate / args.chunk * args.record_second):
            date_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            wave_path = os.path.join(args.output_folder, date_time + args.suffix)
            print("[listen: ] {}".format(wave_path))

            # 保存语音
            wf = wave.open(wave_path, 'wb')
            wf.setnchannels(args.channes)
            wf.setsampwidth(pyaudio_listen.get_sample_size(pyaudio.paInt16))
            wf.setframerate(args.sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            frames = []

    stream.stop_stream()
    stream.close()
    pyaudio_listen.terminate()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mutil Mmicrophone Record Engine')
    parser.add_argument('--output_folder', type=str, default="D:\\temp")
    parser.add_argument('--suffix', type=str, default=".wav")
    parser.add_argument('--chunk', type=int, default=1024)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--channes', type=int, default=6)
    parser.add_argument('--record_second', type=int, default=20)
    args = parser.parse_args()

    listen(args)