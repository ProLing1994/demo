import numpy as np
import sys
import torch 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from TTS.config.vocoder.hparams import *


def infer_waveform(net, mel, batched=True, target=voc_target, overlap=voc_overlap, 
                   progress_callback=None):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    
    :param normalize:  
    :param batched: 
    :param target: 
    :param overlap: 
    :return: 
    """ 
    mel = mel.astype(np.float32) / max_abs_value
    mel = torch.tensor(mel).unsqueeze(0)

    # Inference
    if isinstance(net, torch.nn.parallel.DataParallel):
        wav = net.module.generate(mel, batched, target, overlap, mu_law, progress_callback)
    else:
        wav = net.generate(mel, batched, target, overlap, mu_law, progress_callback)
        
    return wav


    wav_forward = net.generate(mel_forward, bool_gen_batched, target, overlap, mu_law)