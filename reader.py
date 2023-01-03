import numpy as np
import torch
from torch.utils.data import Dataset
import csv
import torchaudio
from data_utils.transform import transform

# from macls.utils.logger import setup_logger

# logger = setup_logger(__name__)


# 音频数据加载器
class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 aug_mode,
                 fea_mode,
                 mode='train',
                 sample_rate=16000,
                 is_mixup=True):

        super(CustomDataset, self).__init__()

        self._target_sample_rate = sample_rate
        self.mode = mode
        self.is_mixup = is_mixup
        self.augment_mode = aug_mode
        self.featurize_mode = fea_mode
        self.device = torch.device("cuda")

        # 获取数据列表
        if mode == "train":
            data_list_path += "train.csv"
        elif mode == "eval":
            data_list_path += "eval.csv"
        elif mode == "dev":
            data_list_path += "dev.csv"

        with open(data_list_path, "r") as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)
            self.lines, self.labels = zip(*lines[1:])

        # self.targets = [
        #     'airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
        #     'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram'
        # ]

        self.tragets = {
            'airport': 0,
            'bus': 1,
            'metro': 2,
            'metro_station': 3,
            'park': 4,
            'public_square': 5,
            'shopping_mall': 6,
            'street_pedestrian': 7,
            'street_traffic': 8,
            'tram': 9
        }

    def __getitem__(self, idx):
        # 读取音频
        data, __ = torchaudio.load(self.lines[idx])
        data = data.to(self.device)

        # 音频增强并转换为频域

        data = transform(data,
                         mode=self.augment_mode,
                         feature_method=self.featurize_mode)

        # One-Hot标签
        label = torch.zeros(10)
        label[self.tragets[self.labels[idx]]] = 1
        label = label.to(self.device)

        # mixup
        if self.mode == "train" and self.is_mixup:
            if (idx > 0 and idx % 5 == 0):
                while True:
                    mixup_idx = np.random.randint(0, len(self.lines) - 1)

                    if self.labels[mixup_idx] != self.labels[idx]:
                        mixup_label = torch.zeros(10)
                        mixup_label[self.tragets[self.labels[mixup_idx]]] = 1
                        mixup_data, __ = torchaudio.load(self.lines[mixup_idx])
                        break

                mixup_label = mixup_label.to(self.device)
                mixup_data = mixup_data.to(self.device)
                mixup_data = transform(mixup_data,
                                       mode=self.augment_mode,
                                       feature_method=self.featurize_mode)

                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                data = lam * data + (1 - lam) * mixup_data
                label = lam * label + (1 - lam) * mixup_label
                # print(label, mixup_label)

        return data, label

    def __len__(self):
        return len(self.lines)
