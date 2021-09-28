### Signal Processing (used in both synthesizer and vocoder)
# Match the values of the synthesizer
min_level_db = -100
ref_level_db = 20
max_abs_value = 4.                         # Gradient explodes if too big, premature convergence if too small.
preemphasis = 0.97                         # Filter coefficient to use if preemphasize is True
preemphasize = True

voc_mode = 'RAW'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode
                                    # below

voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider 
                                    # than input length
voc_seq_multiple = 5                # must be a multiple of hop_length

# Generating / Synthesizing
voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
voc_target = 8000                   # target number of samples to be generated in each batch entry
voc_overlap = 400                   # number of samples for crossfading between batches