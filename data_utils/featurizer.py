import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram, Spectrogram, MFCC


class AudioFeaturizer(nn.Module):
    """音频特征器

    :param feature_conf: 预处理方法的参数
    :type feature_conf: dict
    :param sample_rate: 用于训练的音频的采样率
    :type sample_rate: int
    """
    def __init__(self, feature_method='MelSpectrogram', fs=44100):
        super().__init__()
        wind_dur = 0.04
        wind_shift = 0.02
        window_len = int(wind_dur * fs)
        overlap = int((wind_dur - wind_shift) * fs)

        self._feature_method = feature_method

        if feature_method == 'MelSpectrogram' or feature_method == 'LogMelSpectrogram':
            self.feat_fun = MelSpectrogram(sample_rate=fs,
                                           n_fft=window_len,
                                           hop_length=overlap,
                                           center=False)
        elif feature_method == 'Spectrogram':
            self.feat_fun = Spectrogram(n_fft=window_len,
                                        win_length=window_len,
                                        hop_length=overlap)
        elif feature_method == 'MFCC':
            self.feat_fun = MFCC(n_mfcc=13, sample_rate=fs)

        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')

    def get_feature(self, waveforms):
        """从音频中提取音频特征

        :param waveforms: Audio to extract features from.
        :type waveforms: ndarray
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        feature = self.feat_fun(waveforms)
        feature = feature.transpose(2, 1)
        if self._feature_method == 'LogMelSpectrogram':
            feature = 20 * feature.log10()

        # 归一化
        mean = torch.mean(feature, 1, keepdim=True)
        std = torch.std(feature, 1, keepdim=True)
        feature = (feature - mean) / (std + 1e-5)

        return feature

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self._feature_method == 'LogMelSpectrogram':
            return 128
        elif self._feature_method == 'MelSpectrogram':
            return 128
        elif self._feature_method == 'Spectrogram':
            return 257
        elif self._feature_method == 'MFCC':
            return 13
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))
