import numpy as np
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from TTS.dataset.sv2tts.text import *
from TTS.config.sv2tts.hparams import *
from TTS.dataset.sv2tts.sv2tts_dataset_preload_audio_lmdb import *


def synthesize_spectrograms(cfg, net, texts, embeddings=None):
    # Preprocess text inputs
    inputs = [text_to_sequence(text.strip(), cfg.dataset.tts_cleaner_names) for text in texts]

    # Batch inputs, 16 个放到一个 batch 进行传播
    batched_inputs = [inputs[i: i + synthesis_batch_size]
                            for i in range(0, len(inputs), synthesis_batch_size)]

    if embeddings is not None:
        # Preprocess embeddings inputs
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
            
        batched_embeds = [embeddings[i: i + synthesis_batch_size] for i in range(0, len(embeddings), synthesis_batch_size)]

    specs = []
    for i, batch in enumerate(batched_inputs, 1):
        # Pad texts so they are all the same length
        text_lens = [len(text) for text in batch]
        max_text_len = max(text_lens)
        chars = [pad1d(text, max_text_len) for text in batch]
        chars = np.stack(chars)

        # Stack speaker embeddings into 2D array for batch processing
        speaker_embeds = np.stack(batched_embeds[i-1]) if embeddings is not None else None

        # Convert to tensor
        chars = torch.tensor(chars).long().cuda()
        speaker_embeddings = torch.tensor(speaker_embeds).float().cuda() if embeddings is not None else None

        # Inference
        if isinstance(net, torch.nn.parallel.DataParallel):
            _, mels, _, _ = net.module.inference(chars, speaker_embeddings)
        else:
            _, mels, _, _ = net.inference(chars, speaker_embeddings)

        mels = mels.detach().cpu().numpy()
        for m in mels:
            # Trim silence from end of each spectrogram
            while np.max(m[:, -1]) < tts_stop_threshold:
                m = m[:, :-1]
            specs.append(m)
            
    return specs