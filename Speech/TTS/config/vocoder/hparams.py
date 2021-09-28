### Signal Processing (used in both synthesizer and vocoder)
# Match the values of the synthesizer
min_level_db = -100
ref_level_db = 20
max_abs_value = 4.                         # Gradient explodes if too big, premature convergence if too small.
preemphasis = 0.97                         # Filter coefficient to use if preemphasize is True
preemphasize = True

voc_mode = 'RAW'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode
                                    # below