import librosa
import numpy as np
import sys
import torch 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.dataset import audio 
from Basic.dataset import dataset_augmentation


def embed_frames_batch(frames_batch, cfg, net):
    """
    Computes embeddings for a batch of mel spectrogram.
    
    :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape 
    (batch_size, n_frames, n_channels)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if net is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")
    
    frames = torch.from_numpy(frames_batch).cuda()
    if isinstance(net, torch.nn.parallel.DataParallel):
        if cfg.loss.method == 'embedding':
            embed = net.module.forward(frames).detach().cpu().numpy()
        elif cfg.loss.method == 'classification':
            embed, _ = net.module.forward(frames).detach().cpu().numpy()
    else:
        if cfg.loss.method == 'embedding':
            embed = net.forward(frames).detach().cpu().numpy()
        elif cfg.loss.method == 'classification':
            embed, _ = net.forward(frames).detach().cpu().numpy()

    return embed


def embed_mel_torch(mel_torch, net):
    """
    Computes embeddings for a batch of mel spectrogram.
    
    :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape 
    (batch_size, n_frames, n_channels)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if net is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")
    
    if isinstance(net, torch.nn.parallel.DataParallel):
        embed = net.module.forward(mel_torch)
    else:
        embed = net.forward(mel_torch)

    return embed


def compute_partial_slices(n_samples, cfg, min_pad_coverage=0.75, overlap=0.5):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain 
    partial utterances of <frame_num_realtime> each. Both the waveform and the mel 
    spectrogram slices are returned, so as to make each partial utterance waveform correspond to 
    its spectrogram. This function assumes that the mel spectrogram parameters used are those 
    defined in params_data.py.
    
    The returned ranges may be indexing further than the length of the waveform. It is 
    recommended that you pad the waveform with zeros up to wave_slices[-1].stop.
    
    :param n_samples: the number of samples in the waveform
    :param frame_num_realtime: the number of mel spectrogram frames in each partial 
    utterance
    :param min_pad_coverage: when reaching the last partial utterance, it may or may not have 
    enough frames. If at least <min_pad_coverage> of <frame_num_realtime> are present, 
    then the last partial utterance will be considered, as if we padded the audio. Otherwise, 
    it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial 
    utterance, this parameter is ignored so that the function always returns at least 1 slice.
    :param overlap: by how much the partial utterance should overlap. If set to 0, the partial 
    utterances are entirely disjoint. 
    :return: the waveform slices and mel spectrogram slices as lists of array slices. Index 
    respectively the waveform and the mel spectrogram with these slices to obtain the partial 
    utterances.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1
    
    # 每一帧的帧长
    # frame_num_model_forward：模型前向传播 mel 频率的长度
    # frame_num_realtime：真实时长 mel 频率的长度
    frame_num_model_forward = cfg.dataset.data_size[1]
    frame_num_realtime = int(cfg.dataset.clip_duration_ms / cfg.dataset.window_stride_ms)

    samples_per_frame = int((cfg.dataset.sample_rate * cfg.dataset.window_stride_ms / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(frame_num_realtime * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - frame_num_realtime + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + frame_num_model_forward])
        mel_range_realtime = np.array([i, i + frame_num_realtime])
        wav_range = mel_range_realtime * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))
        
    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]
    
    return wav_slices, mel_slices


def embed_mel(mel_pred, mel_label, mel_length, cfg, sv_net):
    """
    Computes an embedding for mel.
    """ 
    # init 
    batch = len(mel_length)
    mel_pred = mel_pred.permute(0, 2, 1).contiguous() 
    mel_label = mel_label.permute(0, 2, 1).contiguous() 
    
    mel_pred_partial_embeds, mel_label_partial_embeds = [] , []
    for batch_idx in range(batch):
        mel_length_idx = mel_length[batch_idx]
        mel_pred_idx = torch.unsqueeze(mel_pred[batch_idx], dim =0)[:, : mel_length_idx, :]
        mel_label_idx = torch.unsqueeze(mel_label[batch_idx], dim =0)[:, : mel_length_idx, :]
        
        mel_pred_partial_embeds += [ embed_mel_torch(mel_pred_idx, sv_net).squeeze(0) ]
        mel_label_partial_embeds += [ embed_mel_torch(mel_label_idx, sv_net).squeeze(0) ]
        
    mel_pred_partial_embeds = torch.stack(mel_pred_partial_embeds).contiguous()
    mel_label_partial_embeds = torch.stack(mel_label_partial_embeds).contiguous().detach()
    
    return mel_pred_partial_embeds, mel_label_partial_embeds


def embed_utterance(wav, cfg, sv_net):
    """
    Computes an embedding for a single utterance.
    """ 
    # data alignment
    wav = dataset_augmentation.dataset_alignment(cfg, wav, bool_replicate=True)

    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), cfg)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    
    # Split the utterance into partials
    frames = audio.compute_mel_spectrogram(cfg, wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch, cfg, sv_net)
    
    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    if np.linalg.norm(raw_embed, 2) != 0.0:
        raw_embed = raw_embed / np.linalg.norm(raw_embed, 2)
    
    return raw_embed


def embedding(cfg, net, fpath_or_wav):
    ## Computing the embedding
    # First, we load the wav using the function that the speaker encoder provides. This is 
    # important: there is preprocessing that must be applied.

    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str):
        data = librosa.load(str(fpath_or_wav), sr=cfg.dataset.sample_rate)[0]
    else:
        data = fpath_or_wav

    # data trim_silence
    if hparams.trim_silence:
        data = audio.trim_silence_old(data)

    # data rescale
    if hparams.rescale:
        data = data / np.abs(data).max() * hparams.rescaling_max

    # Then we derive the embedding. There are many functions and parameters that the 
    # speaker encoder interfaces. These are mostly for in-depth research. You will typically
    # only use this function (with its default parameters):
    embed = embed_utterance(data, cfg, net)
    # print("Created the embedding")
    return embed