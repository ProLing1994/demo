import numpy as np
import sys
import soundfile as sf
import pysptk as ps
import pyworld as pw

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio
from Basic.dataset.audio import world_feature
from Basic.utils.train_tools import *


def world_feature_extract(cfg, wav_path):
    # params
    minf0 = 1
    maxf0 = 800

    # load wav
    wav, fs = sf.read(wav_path)

    # check sampling frequency
    if not fs == cfg.dataset.sampling_rate:
        raise Exception("ERROR: sampling frequency is not matched.")

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
    log_cont_f0_lpf_range = np.expand_dims(np.log(cont_f0_lpf_range), axis=-1)

    # 编码 AP 非周期序列
    # codeap: 3-dimensional aperiodicity coding coefficien
    codeap_range = pw.code_aperiodicity(ap_range, fs)
    if codeap_range.ndim == 1:
        codeap_range = np.expand_dims(codeap_range, axis=-1)

    # 生成 mcep 倒谱系数
    # mcep：mel-cepstrum coeficient
    mcep_range = ps.sp2mc(spc_range, cfg.dataset.mcep_dim, cfg.dataset.mcep_alpha)

    # 特征拼接 uv 特征 / 对数 F0 基频 / 编码 AP 非周期序列 / mcep 倒谱系数
    feat_org_lf0 = np.c_[uv_range, log_cont_f0_lpf_range, codeap_range, mcep_range]

    return wav, f0_range, spc_range, ap_range, uv_range, log_cont_f0_lpf_range, codeap_range, mcep_range, feat_org_lf0


def world_feature_2wav(cfg, f0, ap, mcep):
    fs = cfg.dataset.sampling_rate

    sp = ps.mc2sp(mcep, cfg.dataset.mcep_alpha, cfg.dataset.fft_size)
    wav_anasyn = np.clip(pw.synthesize(f0, sp, ap, fs, frame_period=cfg.dataset.shiftms), -1, 1)

    return wav_anasyn


def world_feature_2wav_sp(cfg, f0, sp, ap):
    fs = cfg.dataset.sampling_rate
    wav_anasyn = np.clip(pw.synthesize(f0, sp, ap, fs, frame_period=cfg.dataset.shiftms), -1, 1)

    return wav_anasyn


def main():
    # 结果正常
    wav_path = "/mnt/huanyuan2/data/speech/vc/Chinese/vc_test/test/BZNSYP/000491.wav"

    # 结果异常
    # 当噪音明显时，WORLD Feature 不能有效建模
    # wav_path = "/mnt/huanyuan2/data/speech/Recording/RM_Mandarin_Weibo/office/adkit_16k/wav/RM_ROOM_Mandarin_S001M0P1.wav"
    # wav_path = "/mnt/huanyuan2/data/speech/Recording/RM_Mandarin_Weibo/office/android_16k/wav/RM_ROOM_Mandarin_android_S001M0P1.wav"
    config_file = "/home/huanyuan/code/demo/Speech/VC/config/cyclevae/vc_config_chinese_cyclevae.py"    

    cfg = load_cfg_file(config_file)
    wav, f0, sp, ap, uv, log_cont_f0_lpf, codeap, mcep, feat_src = world_feature_extract(cfg, wav_path)

    ap_zero = np.zeros(ap.shape)
    # wav_anasyn = world_feature_2wav(cfg, f0, ap, mcep)
    wav_anasyn = world_feature_2wav_sp(cfg, f0, sp, ap)

    # wav_zero_ap_anasyn = world_feature_2wav(cfg, f0, ap_zero, mcep)
    wav_zero_ap_anasyn = world_feature_2wav_sp(cfg, f0, sp, ap_zero)

    wav_fpath = os.path.join("/home/huanyuan/temp/", "wav_{}.wav".format('world_feature'))
    audio.save_wav(wav_anasyn, str(wav_fpath), sampling_rate=cfg.dataset.sampling_rate)

    wav_fpath = os.path.join("/home/huanyuan/temp/", "wav_{}.wav".format('world_feature_zeroap'))
    audio.save_wav(wav_zero_ap_anasyn, str(wav_fpath), sampling_rate=cfg.dataset.sampling_rate)


if __name__ == "__main__":
    main()