from auditok import split

if __name__ == "__main__":
  audio_path = "/home/huanyuan/data/speech/canting.wav"
  audio_regions = split(audio_path, 2.5, 10, 1, False, True)
  idx = 0
  for region in audio_regions:
    # region.play(progress_bar=True)
    # filename = region.save("/home/huanyuan/data/speech/test/region_{meta.start:.3f}.wav")
    filename = region.save("/home/huanyuan/data/speech/test/RM_ROOM_Mandarin_S001M0P{:0>5d}.wav".format(idx))
    idx += 1
    print("region saved as: {}".format(filename))