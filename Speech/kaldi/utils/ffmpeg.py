import os

def trans_flac_to_wav(flac_path, wav_path):
  command = ['ffmpeg', '-i', flac_path, wav_path]
  command = ' '.join(command)
  print(command)
  os.system(command)

if __name__ == "__main__":
  input_path = "/home/huanyuan/code/kaldi/kaldi/egs/librispeech/online_demo/online-data/audio/"

  flac_list = os.listdir(input_path)
  for idx in range(len(flac_list)):
    if flac_list[idx].endswith('.flac'):
      flac_path = os.path.join(input_path, flac_list[idx]) 
      wav_path = flac_path[:-4] + 'wav'
      trans_flac_to_wav(flac_path, wav_path)
  