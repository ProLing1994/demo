### SV2TTS
silence_min_duration_split = 0.4           # Duration in seconds of a silence for an utterance to be split
utterance_min_duration = 1.6               # Duration in seconds below which utterances are discarded

### infer
synthesis_batch_size = 16                   # For vocoder preprocessing and inference.
tts_stop_threshold = -3.4                   # Value below which audio generation ends.
                                            # For example, for a range of [-4, 4], this
                                            # will terminate the sequence at the first
                                            # frame that has all values < -3.4