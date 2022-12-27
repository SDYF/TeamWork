# import csv
# import torch
# # from feature import get_mel_amp
# import torchaudio
# import numpy as np
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

# from data_utils.logger import setup_logger
from visualdl import LogWriter
from visualdl.server import app

app.run(logdir="log")

# logger = setup_logger(__name__)

# writer = LogWriter(logdir='log_temp')
# for i in range(1, 100):
#     writer.add_scalar('Test/acc', 0.11, i)
#     writer.add_scalar('Train/Loss', 5, i)
