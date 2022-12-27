# from torchaudio.transforms import MelSpectrogram
# from data_utils._awgn import awgn
from data_utils.featurizer import AudioFeaturizer
import torch
import numpy as np

# 对数据先进行增强，再转化为频域


def transform(x, mode=None, feature_method='MelSpectrogram'):

    # 数据增强
    if mode == "origin":
        pass
    if mode == "noise":
        x = awgn(x=x, snr=40)
    else:
        pass

    device = torch.device("cuda")
    # 转换为频谱
    fea = AudioFeaturizer(feature_method=feature_method, fs=44100)
    fea.to(device)
    return fea.get_feature(x)


def awgn(x, snr, seed=7):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    sig = x.data.cpu().numpy()
    np.random.seed(seed)  # 设置随机种子
    snr = 10**(snr / 10.0)
    sig_power = np.sum(sig**2) / len(sig)
    n_power = sig_power / snr
    noise = np.random.randn(len(x)) * np.sqrt(n_power)
    x = torch.from_numpy(sig + noise).to(torch.device("cuda"))
    x = x.to(torch.float32)

    return x