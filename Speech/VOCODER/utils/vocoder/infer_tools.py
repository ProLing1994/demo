import numpy as np
import sys
import torch 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.dataset import audio

import VOCODER.config.vocoder.hparams as hparams_vocoder


def infer_waveform(net, mel, batched=True, target=hparams_vocoder.voc_target, overlap=hparams_vocoder.voc_overlap, 
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
    mel = mel.astype(np.float32) / hparams.max_abs_value
    mel = torch.tensor(mel).unsqueeze(0)

    # Inference
    if isinstance(net, torch.nn.parallel.DataParallel):
        wav = net.module.generate(mel, batched, target, overlap, hparams_vocoder.mu_law, progress_callback)
    else:
        wav = net.generate(mel, batched, target, overlap, hparams_vocoder.mu_law, progress_callback)
        
    return wav

def infer_wavegan(cfg, net, mel, normalize_before=True):
    mel = mel.astype(np.float32).T
    mel = torch.tensor(mel).cuda()

    # Inference
    if isinstance(net, torch.nn.parallel.DataParallel):
        wav = net.module.inference(mel, normalize_before=normalize_before).view(-1)
    else:
        wav = net.inference(mel, normalize_before=normalize_before).view(-1)

    wav = wav.detach().cpu().numpy()
    if cfg.dataset.compute_mel_type == "fbank_preemphasis_log_manual":
        wav = audio.compute_de_emphasis(wav)
    
    return wav
