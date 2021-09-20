### Tacotron Text-to-Speech (TTS)
tts_embed_dims = 512                        # Embedding dimension for the graphemes/phoneme inputs
tts_encoder_dims = 256
tts_decoder_dims = 128
tts_postnet_dims = 512
tts_encoder_K = 5
tts_lstm_dims = 1024
tts_postnet_K = 5
tts_num_highways = 4
tts_dropout = 0.5
tts_cleaner_names = ["english_cleaners"]
tts_stop_threshold = -3.4                   # Value below which audio generation ends.
                                            # For example, for a range of [-4, 4], this
                                            # will terminate the sequence at the first
                                            # frame that has all values < -3.4

tts_clip_grad_norm = 1.0