from auditok import split
import os

if __name__ == "__main__":
  # audio_path = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-103547-114012-000001089960.wav"
  # output_dir = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-103547-114012-000001089960"
  # idx = 12001
  # audio_path = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-131847-141242-000001089960.wav"
  # output_dir = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-131847-141242-000001089960"
  # idx = 12562
  # audio_path = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-145608-152104-000001100420.wav"
  # output_dir = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-145608-152104-000001100420"
  # idx = 13001
  # audio_path = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-152405-152705-000001100420.wav"
  # output_dir = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-152405-152705-000001100420"
  # idx = 13180
  # audio_path = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-152720-160728-000001100420.wav"
  # output_dir = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-152720-160728-000001100420"
  # idx = 13201
  # audio_path = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-160745-162023-000001100420.wav"
  # output_dir = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-160745-162023-000001100420"
  # idx = 13501
  # audio_path = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-162037-163342-000001100420.wav"
  # output_dir = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-162037-163342-000001100420"
  # idx = 13601
  # audio_path = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-163357-164606-000001100420.wav"
  # output_dir = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-163357-164606-000001100420"
  # idx = 13701
  # audio_path = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-164642-170005-000001100420.wav"
  # output_dir = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-164642-170005-000001100420"
  # idx = 13801
  audio_path = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-170017-171407-000001100420.wav"
  output_dir = "/home/huanyuan/data/speech/weiboyulu/1012/0000000000000000-201012-170017-171407-000001100420"
  idx = 13901

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  audio_regions = split(audio_path, 2, 10, 1.5, False, True)
  
  for region in audio_regions:
    # region.play(progress_bar=True)
    # filename = region.save("/home/huanyuan/data/speech/test/region_{meta.start:.3f}.wav")
    filename = region.save(os.path.join(output_dir, "RM_ROOM_Mandarin_S001M0P{:0>5d}.wav".format(idx)))
    idx += 1
    print("region saved as: {}".format(filename))