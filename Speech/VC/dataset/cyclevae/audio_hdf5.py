import librosa
import numpy as np
import sys
import soundfile as sf
import pysptk as ps
import pyworld as pw

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset.audio import world_feature
from Basic.dataset import audio
from Basic.utils.hdf5_tools import *
from Basic.utils.folder_tools import *


def preprocess_audio_normal(cfg, row, hdf5_dir, data_lists):
    # init 
    wav_fpath = row['file']
    spk_name = row['speaker']
    key = row['key']

    # load wav
    wav, fs = sf.read(wav_fpath)
    
    # load conf
    conf_path = os.path.join(hdf5_dir, 'conf', spk_name + '.f0')
    with open(conf_path, "r") as f :
        line = f.readline().strip()
        minf0 = float(line.split(' ')[0])
        maxf0 = float(line.split(' ')[1])

    conf_path = os.path.join(hdf5_dir, 'conf', spk_name + '.pow')
    with open(conf_path, "r") as f :
        line = f.readline().strip()
        pow = float(line.split(' ')[0])
    
    # check sampling frequency
    if not fs == cfg.dataset.sampling_rate:
        raise Exception("ERROR: sampling frequency is not matched.")

    # check mel type
    if not cfg.dataset.compute_mel_type == "world":
        raise Exception("ERROR: mel type is not matched.")

    # highpass cutoff
    highpass_cutoff = cfg.dataset.highpass_cutoff
    if highpass_cutoff != 0:
        wav = world_feature.low_cut_filter(wav, fs, highpass_cutoff)

    # WORLD 计算 F0 基频 / SP 频谱包络 / AP 非周期序列
    _, f0_range, spc_range, ap_range = world_feature.analyze_range(wav, fs=fs, minf0=minf0, \
                                                        maxf0=maxf0, fperiod=cfg.dataset.shiftms, fftl=cfg.dataset.fft_size)

    # 生成 uv 特征 / 连续 F0 基频
    # uv: an unvoiced/voiced binary decision feature
    uv_range, cont_f0_range = world_feature.convert_continuos_f0(np.array(f0_range))
    uv_range = np.expand_dims(uv_range, axis=-1)

    # 低通滤波器修正 F0 基频
    # F0: log-scaled of continuous F0
    cont_f0_lpf_range = world_feature.low_pass_filter(cont_f0_range, int(1.0 / (cfg.dataset.shiftms * 0.001)), cutoff=world_feature.LOWPASS_CUTOFF)
    cont_f0_lpf_range = np.expand_dims(cont_f0_lpf_range, axis=-1)

    # 编码 AP 非周期序列
    # codeap: 3-dimensional aperiodicity coding coefficien
    codeap_range = pw.code_aperiodicity(ap_range, fs)
    if codeap_range.ndim == 1:
        codeap_range = np.expand_dims(codeap_range, axis=-1)

    # 生成 mcep 倒谱系数
    # mcep：mel-cepstrum coeficient
    mcep_range = ps.sp2mc(spc_range, cfg.dataset.mcep_dim, cfg.dataset.mcep_alpha)

    # 特征拼接 uv 特征 / 对数 F0 基频 / 编码 AP 非周期序列 / mcep 倒谱系数
    feat_org_lf0 = np.c_[uv_range, np.log(cont_f0_lpf_range), codeap_range, mcep_range]

    # 通过 SP 频谱包络计算功率
    npow_range = world_feature.spc2npow(spc_range)

    # 计算有效帧位置
    mcepspc_range, spcidx_range = world_feature.extfrm(mcep_range, npow_range, power_threshold=pow)

    # save hdf5
    key_name = key.split('.')[0]
    output_dir = os.path.join(hdf5_dir, 'world')
    create_folder(output_dir)
    
    write_hdf5(os.path.join(output_dir, f"{key_name}.h5"), "wave", wav.astype(np.float32))
    write_hdf5(os.path.join(output_dir, f"{key_name}.h5"), "f0_range", f0_range)
    write_hdf5(os.path.join(output_dir, f"{key_name}.h5"), "feat_org_lf0", feat_org_lf0)
    write_hdf5(os.path.join(output_dir, f"{key_name}.h5"), "npow_range", npow_range)
    write_hdf5(os.path.join(output_dir, f"{key_name}.h5"), "spcidx_range", spcidx_range)

    # save wav
    wav_filt_dir = os.path.join(hdf5_dir, 'wav_filtered')
    create_folder(wav_filt_dir)
    if highpass_cutoff != 0 and wav_filt_dir is not None:
        audio.save_wav(wav, os.path.join(wav_filt_dir, key), fs)

    # save wav_anasyn
    wav_anasyn_dir = os.path.join(hdf5_dir, 'wav_anasyn')
    create_folder(wav_anasyn_dir)
    sp_rec = ps.mc2sp(mcep_range, cfg.dataset.mcep_alpha, cfg.dataset.fft_size)
    wav_anasyn = np.clip(pw.synthesize(f0_range, sp_rec, ap_range, fs, frame_period=cfg.dataset.shiftms), \
                    -1, 1)
    audio.save_wav(wav_anasyn, os.path.join(wav_anasyn_dir, key), fs)

    # data_lists
    data_lists.append({'dataset': row['dataset'], 'speaker': row['speaker'], 'section': row['section'], \
                        'utterance': row['utterance'], 'key': key, 'file': row['file'], \
                        'state': os.path.join(output_dir, f"{key_name}.h5"), 'mode': row['mode']})

    return


def preprocess_audio_hdf5(cfg, row, dataset_name, hdf5_dir, data_lists):
    
    if dataset_name in ['VCC2020']: 
        preprocess_audio_normal(cfg, row, hdf5_dir, data_lists)

    return 