import math
import numpy as np


def label_2_float(x, bits) :
    return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits) :
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)


def encode_mu_law(x, mu) :
    '''
    μ 律压缩：y=ln(1+μx)/ln(1+μ）
    其中：x 为归一化的量化器输入, 值域 [-1, 1], y 为归一化的量化器输出, 值域 [-1, 1]。常数 μ 愈大，则小信号的压扩效益愈高，目前多采用 μ = 255
    作用：改善小信号信噪比，对大信号进行压缩而对小信号进行较大的放大
    https://baike.baidu.com/item/mu-law/4857952?fr=aladdin

    np.floor((fx + 1) / 2 * mu + 0.5)
    将值域 [-1, 1] 的输入，量化到 [0, mu] 中，此处 mu = 511
    '''
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True) :
    if from_labels: 
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x