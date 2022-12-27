import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.panns import CNN6, CNN10, CNN14
from reader import CustomDataset
from sklearn.metrics import confusion_matrix
from data_utils.utils import plot_confusion_matrix
from data_utils.logger import setup_logger

logger = setup_logger(__name__)


class Predicter:
    def __init__(self, use_gpu, fea_mode, aug_mode, batch_size, max_epoch, use_model,
                 model_dir, input_size):
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")

        self.use_gpu = use_gpu
        self.model = None
        self.predict_dataloder = None
        self.class_labels = [
            'airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
            'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram'
        ]
        self.filename = r"Dcase2020task1subset/"
        self.fea_mode = fea_mode
        self.aug_mode = aug_mode
        self.model = None
        self.input_size = input_size

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.use_model = use_model
        self.model_path = model_dir

    def setup_dataloader(self):

        self.predict_dataset = CustomDataset(data_list_path=self.filename,
                                             aug_mode=self.aug_mode,
                                             fea_mode=self.fea_mode,
                                             mode="eval",
                                             sample_rate=44100,
                                             is_mixup=False)
        self.predict_dataloder = DataLoader(dataset=self.predict_dataset,
                                            batch_size=self.batch_size,
                                            sampler=None)

    def setup_model(self, input_size, is_train=False):
        # 获取模型

        if self.use_model == 'panns_cnn6':
            self.model = CNN6(input_size=input_size, num_class=10)
        elif self.use_model == 'panns_cnn10':
            self.model = CNN10(input_size=input_size, num_class=10)
        elif self.use_model == 'panns_cnn14':
            self.model = CNN14(input_size=input_size, num_class=10)
        else:
            raise Exception(f'{self.use_model} 模型不存在！')
        self.model.to(self.device)

        if os.path.isdir(self.model_path):
            self.model_path = os.path.join(self.model_path, 'inference.pt')
        assert os.path.exists(self.model_path), f"{self.model_path} 模型不存在！"
        if torch.cuda.is_available() and self.use_gpu:
            model_state_dict = torch.load(self.model_path)
        else:
            model_state_dict = torch.load(self.model_path, map_location='cpu')
        self.model.load_state_dict(model_state_dict)
        print(f"成功加载模型参数：{self.model_path}")
        self.model.eval()

    def predict_all(self, save_matrix_path):
        self.setup_dataloader()
        self.setup_model(input_size=self.input_size, is_train=False)
        predict_model = self.model

        accuracies, preds, labels = [], [], []
        with torch.no_grad():
            for batch_id, (audio, label) in enumerate(tqdm(self.predict_dataloder)):
                output = predict_model(audio)

                label_temp = label.data.cpu().numpy()
                label_temp = np.argmax(label_temp, axis=1)
                output = output.data.cpu().numpy()
                # 模型预测标签
                pred = np.argmax(output, axis=1)
                preds.extend(pred.tolist())
                # 真实标签
                labels.extend(label_temp.tolist())

                # 计算准确率

                acc = np.mean((pred == label_temp).astype(int))
                accuracies.append(acc)

        acc = float(sum(accuracies) / len(accuracies))
        # 保存混合矩阵
        if save_matrix_path is not None:
            cm = confusion_matrix(labels, preds)
            martix_name = f'{use_model}_{fea_mode}_{aug_mode}/' + 'predict' + '.png'
            plot_confusion_matrix(cm=cm,
                                  save_path=os.path.join(save_matrix_path,
                                                         martix_name),
                                  class_labels=self.class_labels)

        self.model.train()
        print("\n")
        print('正确率为:', round(acc * 100, 3), '%')


if __name__ == "__main__":
    save_model_path = 'models/'

    batch_size = 32
    max_epoch = 30
    fea_mode_all = ('LogMelSpectrogram', 'MelSpectrogram', 'MFCC', 'Spectrogram')
    aug_mode_all = ('origin', 'noise')
    use_model_all = ('panns_cnn6', 'panns_cnn10', 'panns_cnn14')
    input_size = {
        'LogMelSpectrogram': 128,
        'MelSpectrogram': 128,
        'MFCC': 13,
        'Spectrogram': 257
    }

    fea_mode = fea_mode_all[1]
    aug_mode = aug_mode_all[0]
    use_model = use_model_all[0]
    learning_rate = 0.001

    model_dir = os.path.join(save_model_path, f'{use_model}_{fea_mode}_{aug_mode}/')
    print(model_dir)
    my_predict = Predicter(use_gpu=True,
                           fea_mode=fea_mode,
                           aug_mode=aug_mode,
                           batch_size=batch_size,
                           max_epoch=max_epoch,
                           use_model=use_model,
                           model_dir=model_dir,
                           input_size=input_size[fea_mode])

    matrix_path = "predict_matrix/"

    my_predict.predict_all(save_matrix_path=matrix_path)
