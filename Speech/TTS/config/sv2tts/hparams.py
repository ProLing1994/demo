
### 数据预处理阶段
## Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
# 在数据预处理阶段，对过长的数据进行截断，数据过长则不参与后续训练过程
# train samples of lengths between 3sec and 14sec are more than enough to make a model capable
# of good parallelization.
clip_mels_length = True                     # If true, discards samples exceeding max_mel_frames
max_mel_frames = 900    

### SV2TTS
silence_min_duration_split = 0.4           # Duration in seconds of a silence for an utterance to be split
utterance_min_duration = 1.6               # Duration in seconds below which utterances are discarded

### infer
synthesis_batch_size = 16                   # For vocoder preprocessing and inference.
tts_stop_threshold = -3.4                   # Value below which audio generation ends.
                                            # For example, for a range of [-4, 4], this
                                            # will terminate the sequence at the first
                                            # frame that has all values < -3.4