import numpy as np
import pysptk as ps
import pyworld as pw
from scipy.interpolate import interp1d
from scipy.signal import firwin
from scipy.signal import lfilter

FS = 24000
MINF0 = 40
MAXF0 = 700
SHIFTMS = 5
FFTL = 2048
LOWPASS_CUTOFF = 20
HIGHPASS_CUTOFF = 65


def low_cut_filter(x, fs, cutoff=HIGHPASS_CUTOFF):
    """FUNCTION TO APPLY LOW CUT FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(1023, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def low_pass_filter(x, fs, cutoff=LOWPASS_CUTOFF, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x


def analyze(wav, fs=FS, minf0=MINF0, maxf0=MAXF0, fperiod=SHIFTMS, fftl=FFTL, f0=None, time_axis=None):
    if f0 is None or time_axis is None:
        _f0, time_axis = pw.harvest(wav, fs, f0_floor=60.0, frame_period=fperiod)
        f0 = pw.stonemask(wav, _f0, time_axis, fs) 
    sp = pw.cheaptrick(wav, f0, time_axis, fs, fft_size=fftl)
    ap = pw.d4c(wav, f0, time_axis, fs, fft_size=fftl)

    return time_axis, f0, sp, ap



def analyze_range(wav, fs=FS, minf0=MINF0, maxf0=MAXF0, fperiod=SHIFTMS, fftl=FFTL, f0=None, time_axis=None):
    if f0 is None or time_axis is None:
        _f0, time_axis = pw.harvest(wav, fs, f0_floor=minf0, f0_ceil=maxf0, frame_period=fperiod)
        f0 = pw.stonemask(wav, _f0, time_axis, fs) 
    sp = pw.cheaptrick(wav, f0, time_axis, fs, fft_size=fftl)
    ap = pw.d4c(wav, f0, time_axis, fs, fft_size=fftl)

    return time_axis, f0, sp, ap


def spc2npow(spectrogram):
    npow = np.apply_along_axis(spvec2pow, 1, spectrogram)

    meanpow = np.mean(npow)
    npow = 10.0 * np.log10(npow/meanpow)

    return npow


def spvec2pow(specvec):
    fftl2 = len(specvec) - 1
    fftl = fftl2 * 2

    power = specvec[0] + specvec[fftl2]
    for k in range(1, fftl2):
        power += 2.0 * specvec[k]
    power /= fftl

    return power


def extfrm(data, npow, power_threshold=-20):
    T = data.shape[0]
    if T != len(npow):
        raise("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata, valid_index


def convert_f0(f0, f0_mean_src, f0_std_src, f0_mean_trg, f0_std_trg):
    nonzero_indices = f0 > 0
    cvf0 = np.zeros(len(f0))
    cvf0[nonzero_indices] = \
            np.exp((f0_std_trg/f0_std_src)*(np.log(f0[nonzero_indices])-f0_mean_src)+f0_mean_trg)

    return cvf0


def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0