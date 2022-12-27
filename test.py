# import csv
import torch
# # from feature import get_mel_amp
# import torchaudio
import numpy as np
# from torchaudio.transforms import MelSpectrogram

# data_filename = r"Dcase2020task1subset/"
# with open(data_filename + "dev.csv", "r") as f:
#     reader = csv.reader(f, delimiter=',')
#     lines = list(reader)
# idxs, labels = zip(*lines[1:])
# dev_idxs = idxs[0:9]

# unique_labels = [
#     'airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
#     'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram'
# ]
# num_classes = len(unique_labels)

# wind_dur = 0.04
# wind_shift = 0.02
# for filename in dev_idxs:
#     data, fs = torchaudio.load(filename)
#     print(data)
#     window_len = int(wind_dur * fs)
#     overlap = int((wind_dur - wind_shift) * fs)
#     mel = MelSpectrogram(sample_rate=fs,
#                          n_fft=window_len,
#                          hop_length=overlap,
#                          center=False)(data)
#     print(fs)
#     print(mel.transpose(2, 1).shape)

#     break

from reader import CustomDataset

if __name__ == "__main__":
    fea_mode_all = ('LogMelSpectrogram', 'MelSpectrogram', 'Spectrogram', 'MFCC',
                    'Delta', 'Delta-Delta')
    aug_mode_all = ('origin', 'noise')

    filename = r"Dcase2020task1subset/"

    fea_mode = fea_mode_all[5]
    aug_mode = aug_mode_all[0]

    dataset = CustomDataset(data_list_path=filename,
                            aug_mode=aug_mode,
                            fea_mode=fea_mode,
                            mode='eval',
                            sample_rate=44100)

    data, __ = dataset.__getitem__(idx=0)
    print(data.shape)
    # a = ('MFCC', 'Delta', 'Delta-Delta')
    # b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # print(fea_mode in a)
    # print(b[-2:])

# a = np.array([[-0.4025, -0.5222, -0.8394, -0.1124, -0.2314],
#               [-0.2089, -0.1762, -0.2196, -0.1223, -0.4347],
#               [-0.7841, -0.4364, -0.1739, -0.5013, -0.1215],
#               [-0.6259, -0.7144, 0.0000, -0.1024, -0.2826],
#               [-0.7223, -0.1515, -0.6154, -0.4741, -0.2241],
#               [0.2, 0.3, 0.4, 0.5, 0.6]])
# a = torch.from_numpy(a)
# b = a.max(dim=1)[0]
# c = a.permute(1, 0)
# # print(b)
# # print((c - b).permute(1, 0))
# d = a[1]
# e = d - d.max()

# print(d, e)
# f = torch.softmax(d, dim=0)
# print(torch.softmax(f, dim=0))
# print(torch.softmax(f, dim=0))
