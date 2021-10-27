import librosa
import numpy as np
import random
import struct
import sys
import webrtcvad

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech/')
sys.path.insert(0, '/home/huanyuan/code/demo/Speech')

from KWS.config.kws import hparams


def dataset_alignment(cfg, data, bool_replicate=False):
    # init
    desired_samples = int(cfg.dataset.sample_rate * cfg.dataset.clip_duration_ms / 1000)

    # alignment data
    if len(data) < desired_samples:
        data_length = len(data)
        if bool_replicate:
            tile_size = (desired_samples // data_length) + 1
            data = np.tile(data, tile_size)[:desired_samples]
        else:
            data_offset = np.random.randint(0, desired_samples - data_length)
            data = np.pad(data, (data_offset, 0), "constant")
            data = np.pad(data, (0, desired_samples - data_length - data_offset), "constant")

    if len(data) > desired_samples:
        data_offset = np.random.randint(0, len(data) - desired_samples)
        data = data[data_offset:(data_offset + desired_samples)]

    assert len(data) == desired_samples, "[ERROR:] Something wronge with audio length, please check"
    return data


def dataset_augmentation_volume_speed(cfg, data):
    # init
    speed_list = cfg.dataset.augmentation.speed
    volume_list = cfg.dataset.augmentation.volume

    speed = np.random.uniform(speed_list[0], speed_list[1])  
    volume = np.random.uniform(volume_list[0], volume_list[1])  
    
    # speed > 1, 加快速度
    # speed < 1, 放慢速度
    data = librosa.effects.time_stretch(data, speed)

    # 音量大小调节
    data = data * volume
    return data


def dataset_augmentation_pitch(cfg, data):
    # init
    pitch_list = cfg.dataset.augmentation.pitch 

    pitch = np.random.randint(pitch_list[0], pitch_list[1])  

    # 音调调节
    data = librosa.effects.pitch_shift(data, sr=cfg.dataset.sample_rate, n_steps=pitch)
    return data


def dataset_augmentation_time_shift(cfg, data, bool_time_shift_multiple=False):
    time_shift_ms = cfg.dataset.augmentation.time_shift_ms
    time_shift_samples = int(cfg.dataset.sample_rate * time_shift_ms / 1000)
    
    # Time shift enhancement multiple of negative samples,
    # which is effective for advanced prediction and lag prediction
    if bool_time_shift_multiple:
        time_shift_multiple = cfg.dataset.augmentation.time_shift_multiple
        time_shift_samples *= time_shift_multiple

    if time_shift_samples > 0:
        time_shift_amount = np.random.randint(-time_shift_samples, time_shift_samples)
        time_shift_left = -min(0, time_shift_amount)
        time_shift_right = max(0, time_shift_amount)
        data = np.pad(data, (time_shift_left, time_shift_right), "constant")
        data = data[:len(data) - time_shift_left] if time_shift_left else data[time_shift_right:]
    return data


def dataset_add_synthetic_noise(cfg, data):
    # init
    synthetic_type = cfg.dataset.augmentation.synthetic_type
    synthetic_frequency = cfg.dataset.augmentation.synthetic_frequency
    synthetic_scale = cfg.dataset.augmentation.synthetic_scale
    synthetic_prob = cfg.dataset.augmentation.synthetic_prob

    if not synthetic_frequency > 0:
        return data
        
    if np.random.uniform(0, 1) < synthetic_frequency:
        if synthetic_type == "white":
            scale = synthetic_scale * np.random.uniform(0, 1)
            synthetic_noise = np.random.normal(loc=0, scale=scale, size=data.shape)
            data = (data + synthetic_noise)
        elif synthetic_type == "salt_pepper": 
            prob = synthetic_prob * np.random.uniform(0, 1)
            synthetic_noise = np.random.binomial(1, prob, size=data.shape) - np.random.binomial(1, prob, size=data.shape) 
            data = (data + synthetic_noise)

    # data clip
    data = np.clip(data, -1.0, 1.0)
    return data


def dataset_add_noise(cfg, data, background_data, bool_force_add_noise=False):          
    # init
    background_frequency = cfg.dataset.augmentation.background_frequency 
    background_volume = cfg.dataset.augmentation.background_volume

    if not background_frequency > 0:
        return data
    
    assert len(background_data) > 0, "[ERROR:] Something wronge with background data, please check"

    background_idx = np.random.randint(len(background_data))
    background_sample = background_data[background_idx]

    # alignment background data
    background_clipped = dataset_alignment(cfg, background_sample, True) 

    if np.random.uniform(0, 1) < background_frequency or bool_force_add_noise:
        background_volume = np.random.uniform(0, background_volume)

    data_max_value = data.max()
    background_max_value = (background_volume * background_clipped).max() * 0.8
    if background_max_value < data_max_value or bool_force_add_noise:
        data = background_volume * background_clipped + data

    # add synthetic noise
    data = dataset_add_synthetic_noise(cfg, data)

    # data clip
    data = np.clip(data, -1.0, 1.0)
    return data


def dataset_augmentation_vad(cfg, data):
    # init
    sample_rate = cfg.dataset.sample_rate
    vad_frequency = cfg.dataset.augmentation.vad_frequency 
    vad_window_length = random.choice(cfg.dataset.augmentation.vad_window_length) 
    vad_mode = random.choice(cfg.dataset.augmentation.vad_mode) 
    vad_chunk_size = (vad_window_length * sample_rate) // 1000
    
    if np.random.uniform(0, 1) >= vad_frequency:
        return data

    # Trim the end of the audio to have a multiple of the window size
    data = data[:len(data) - (len(data) % vad_chunk_size)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(data), *(np.round(data * hparams.int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    vad = webrtcvad.Vad(mode=vad_mode)
    vad_flags = []
    for window_start in range(0, len(data), vad_chunk_size):
        window_end = window_start + vad_chunk_size
        vad_flags.append(vad.is_speech(pcm_wave[window_start * 2 : window_end * 2],
                                         sample_rate=sample_rate))
    
    for vad_idx in range(len(vad_flags)):
        if vad_flags[vad_idx] == 0:
            window_start = vad_idx * vad_chunk_size
            window_end = window_start + vad_chunk_size
            data[window_start: window_end] = 0
            
    return data


def dataset_augmentation_waveform(cfg, data, background_data, bool_time_shift_multiple=False):
    # data augmentation
    if cfg.dataset.augmentation.speed_volume_on:
        data = dataset_augmentation_volume_speed(cfg, data)
    
    # data augmentation
    if cfg.dataset.augmentation.pitch_on:
        data = dataset_augmentation_pitch(cfg, data)

    # data alignment
    data = dataset_alignment(cfg, data) 

    # add time_shift
    data = dataset_augmentation_time_shift(cfg, data, bool_time_shift_multiple)

    # add noise
    data = dataset_add_noise(cfg, data, background_data)

    # vad augmentation
    if cfg.dataset.augmentation.vad_on:
        data = dataset_augmentation_vad(cfg, data)

    # data alignment
    data = dataset_alignment(cfg, data) 
    return data


def add_frequence_mask(spec, F=27, num_masks=1, replace_with_zero=False, static=False):
    num_mel_channels = spec.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)

        if static:
            f = F

        f_start = random.randrange(0, num_mel_channels - f)
        f_end = f_start + f

        # avoids randrange error if values are equal and range is empty
        if (f_start == f_start + f):
            return spec

        if (replace_with_zero):
            spec[:, f_start:f_end] = 0
        else:
            spec[:, f_start:f_end] = spec.mean()

    return spec


def add_time_mask(spec, T=40, num_masks=1, replace_with_zero=False, static=False):
    len_spectro = spec.shape[0]
    
    for i in range(0, num_masks):
        t = random.randrange(0, T)

        if static:
            t = T

        t_start = random.randrange(0, len_spectro - t)
        t_end = t_start + t

        # avoids randrange error if values are equal and range is empty
        if (t_start == t_start + t):
            return spec

        if (replace_with_zero): 
            spec[t_start:t_end] = 0
        else: 
            spec[t_start:t_end] = spec.mean()

    return spec


def dataset_augmentation_spectrum(cfg, audio_data):
    # init
    F = cfg.dataset.augmentation.F
    T = cfg.dataset.augmentation.T
    num_masks = cfg.dataset.augmentation.num_masks

    # add SpecAugment
    if cfg.dataset.augmentation.spec_on:
        audio_data = add_frequence_mask(audio_data, F=F, num_masks=num_masks, replace_with_zero=True)
        audio_data = add_time_mask(audio_data, T=T, num_masks=num_masks, replace_with_zero=True)
    return audio_data