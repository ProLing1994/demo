import numpy as np
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from TTS.dataset.text.text import *
from TTS.config.sv2tts.hparams import *
from TTS.dataset.sv2tts.sv2tts_dataset_preload_audio_lmdb import *

def self_cor(mat) : 
    mode = np.sqrt(np.sum(mat**2, axis=0))[:, None]
    return mat.T @ mat / (mode.T * mode)

def align_measure(attn_mat) : 

    """
    measures an attention map's `correctness`;
    a large value means the attention scores are scattered and incorrect,
    while a small value indicates the alignment tends to form a diagonal-line.

    the measure is length-normalized, but the final silent segments could cause
    the result to raise, but because the model tends to synthesize a constant length
    silent ending segment, this effect could only result in a constant offset from zero
    when comparing among results that are about the same lengths (usually ~ 10)

    preliminary experiments shows : 
        10 usually means good;
        30 means the diagonal trends does exist, but about half of the attetion-map is scatterd
        60 or higher means completely lose focus
    """

    L = attn_mat.shape[0]
    Q = self_cor(attn_mat.T)
    return (np.trace(Q @ Q.T) - L) / L

def synthesize_spectrogram(cfg, net, text, embedding=None):
    # Preprocess text inputs
    char = text_to_sequence(text.strip(), cfg.dataset.tts_cleaner_names, lang=cfg.dataset.symbols_lang)
    char = np.stack(char)

    # Convert to tensor
    char = torch.tensor(char).long().cuda()
    char = torch.unsqueeze(char, dim=0)

    # Stack speaker embeddings into 2D array for batch processing
    speaker_embedding = None
    if not embedding is None:
        speaker_embedding = np.stack(embedding)
        speaker_embedding = torch.tensor(speaker_embedding).float().cuda()
        speaker_embedding = torch.unsqueeze(speaker_embedding, dim=0)

    # Inference
    if isinstance(net, torch.nn.parallel.DataParallel):
        _, mels, _, attention = net.module.inference(char, speaker_embedding)
    else:
        _, mels, _, attention = net.inference(char, speaker_embedding)

    print(f'align score : {align_measure(attention.cpu().detach().numpy()[0].T)}')
    mels = mels.detach().cpu().numpy()
    for m in mels:
        # Trim silence from end of each spectrogram
        while np.max(m[:, -1]) < tts_stop_threshold:
            m = m[:, :-1]
            
    return m