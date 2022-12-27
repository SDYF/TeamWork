import time
import os
from datetime import timedelta
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary
from models.panns import CNN6, CNN10, CNN14
from reader import CustomDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from data_utils.utils import plot_confusion_matrix

from visualdl import LogWriter
from data_utils.logger import setup_logger

logger = setup_logger(__name__)


class Trainer():
    def __init__(self, use_gpu, fea_mode, aug_mode, batch_size, max_epoch, use_model,
                 learning_rate, input_size):
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")

        self.use_gpu = use_gpu
        self.model = None
        self.dev_dataloder = None
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
        self.learning_rate = learning_rate
        self.dev_acc = 0.0
        self.lower_than_best = int(0)

    def setup_dataloader(self, is_train=False):
        if is_train:
            self.train_dataset = CustomDataset(data_list_path=self.filename,
                                               aug_mode=self.aug_mode,
                                               fea_mode=self.fea_mode,
                                               mode="train",
                                               sample_rate=44100,
                                               is_mixup=True)
            self.train_dataloder = DataLoader(dataset=self.train_dataset,
                                              shuffle=True,
                                              batch_size=self.batch_size,
                                              sampler=None)
        else:
            self.dev_dataset = CustomDataset(data_list_path=self.filename,
                                             aug_mode=self.aug_mode,
                                             fea_mode=self.fea_mode,
                                             mode="dev",
                                             sample_rate=44100,
                                             is_mixup=False)
            self.dev_dataloder = DataLoader(dataset=self.dev_dataset,
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

        # print(self.model)
        # 获取损失函数
        self.loss = torch.nn.CrossEntropyLoss()
        if is_train:
            # 获取优化方法
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=float(self.learning_rate),
                                              weight_decay=float(1e-6))
            # 学习率衰减函数
            self.scheduler = CosineAnnealingLR(self.optimizer,
                                               T_max=int(self.max_epoch * 1.2))

    # 保存模型
    def save_checkpoint(self,
                        save_model_path,
                        epoch_id,
                        best_acc=0.,
                        best_model=False):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        if best_model:
            model_path = os.path.join(
                save_model_path, f'{self.use_model}_{self.fea_mode}_{self.aug_mode}',
                'best_model')
        else:
            model_path = os.path.join(
                save_model_path, f'{self.use_model}_{self.fea_mode}_{self.aug_mode}',
                'epoch_{}'.format(epoch_id))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.optimizer.state_dict(),
                   os.path.join(model_path, 'optimizer.pt'))
        torch.save(state_dict, os.path.join(model_path, 'model.pt'))
        with open(os.path.join(model_path, 'model.state'), 'w',
                  encoding='utf-8') as f:
            f.write('{"last_epoch": %d, "accuracy": %f}' % (epoch_id, best_acc))
        if not best_model:
            last_model_path = os.path.join(
                save_model_path, f'{self.use_model}_{self.fea_mode}_{self.aug_mode}',
                'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(
                save_model_path, f'{self.use_model}_{self.fea_mode}_{self.aug_mode}',
                'epoch_{}'.format(epoch_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('已保存模型：{}'.format(model_path))

    def train_epoch(self, epoch_id, writer):
        train_times, accuracies, loss_sum = [], [], []
        start = time.time()
        sum_batch = len(self.train_dataloder) * self.max_epoch
        for batch_id, (audio, label) in enumerate(self.train_dataloder):
            output = self.model(audio)
            # output = torch.nn.functional.softmax(output, dim=-1)
            # print(output)
            # input()
            # 计算loss
            los = self.loss(output, label)
            self.optimizer.zero_grad()
            los.backward()
            self.optimizer.step()

            # 计算准确率
            output = torch.nn.functional.softmax(output, dim=-1)

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label_temp = label.data.cpu().numpy()
            label_temp = np.argmax(label_temp, axis=1)
            acc = np.mean((output == label_temp).astype(int))
            accuracies.append(acc)
            loss_sum.append(los)
            train_times.append((time.time() - start) * 1000)

            # 打印训练状态
            if batch_id % 10 == 0:
                # 计算每秒训练数据量
                train_speed = self.batch_size / (sum(train_times) /
                                                 len(train_times) / 1000)
                # 计算剩余时间
                eta_sec = (sum(train_times) / len(train_times)) * (
                    sum_batch -
                    (epoch_id - 1) * len(self.train_dataloder) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(
                    f'Train epoch: [{epoch_id}/{self.max_epoch}], '
                    f'batch: [{batch_id}/{len(self.train_dataloder)}], '
                    f'loss: {sum(loss_sum) / len(loss_sum):.5f}, '
                    f'accuracy: {sum(accuracies) / len(accuracies):.5f}, '
                    f'learning rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                    f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss',
                                  sum(loss_sum) / len(loss_sum), self.train_step)
                writer.add_scalar('Train/Accuracy',
                                  (sum(accuracies) / len(accuracies)),
                                  self.train_step)
                self.train_step += 1
                train_times = []
            start = time.time()
        self.scheduler.step()

    def evaluate(self, resume_model, save_matrix_path=None):
        """
        评估模型
        :param resume_model: 所使用的模型
        :param save_matrix_path: 保存混合矩阵的路径
        :return: 评估结果
        """
        if self.dev_dataloder is None:
            self.setup_dataloader()
        if self.model is None:
            self.setup_model(input_size=self.input_size)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pt')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = torch.load(resume_model)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            dev_model = self.model.module
        else:
            dev_model = self.model

        accuracies, losses, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (audio, label) in enumerate(tqdm(self.dev_dataloder)):
                output = dev_model(audio)
                los = self.loss(output, label)
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
                losses.append(los.data.cpu().numpy())
        loss = float(sum(losses) / len(losses))
        acc = float(sum(accuracies) / len(accuracies))
        # 保存混合矩阵
        if save_matrix_path is not None:
            cm = confusion_matrix(labels, preds)
            plot_confusion_matrix(cm=cm,
                                  save_path=os.path.join(save_matrix_path,
                                                         f'{int(time.time())}.png'),
                                  class_labels=self.class_labels)

        self.model.train()
        return loss, acc

    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None):

        logdir = os.path.join('log/',
                              f'{self.use_model}_{self.fea_mode}_{self.aug_mode}')
        writer = LogWriter(logdir=logdir)
        self.setup_dataloader(is_train=True)
        self.setup_model(input_size=self.input_size, is_train=True)
        logger.info('训练数据：{}'.format(len(self.train_dataset)))

        dev_step, self.train_step = 0, 0
        best_acc = 0

        for epoch_id in range(self.max_epoch):
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            self.train_epoch(epoch_id=epoch_id, writer=writer)

            # 输出信息
            logger.info('=' * 120)
            save_martix_path = os.path.join(
                'matrix/', f'{self.use_model}_{self.fea_mode}_{self.aug_mode}/')
            loss, acc = self.evaluate(resume_model=None,
                                      save_matrix_path=save_martix_path)
            if acc < self.dev_acc:
                if epoch_id >= 20:
                    self.lower_than_best += 1
                    if self.lower_than_best >= 10:
                        logger.info("验证集准确率下降，迭代终止！")
                        return
            else:
                self.dev_acc = acc
                self.lower_than_best = 0

            logger.info(
                'Dev epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}'.
                format(epoch_id, str(timedelta(seconds=(time.time() - start_epoch))),
                       loss, acc))
            logger.info('=' * 120)
            writer.add_scalar('Dev/Accuracy', acc, dev_step)
            writer.add_scalar('Dev/Loss', loss, dev_step)
            dev_step += 1
            self.model.train()
            # 记录学习率
            writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], epoch_id)
            # # 保存最优模型
            if acc >= best_acc:
                best_acc = acc
                self.save_checkpoint(save_model_path=save_model_path,
                                     epoch_id=epoch_id,
                                     best_acc=acc,
                                     best_model=True)
            # 保存模型
            self.save_checkpoint(save_model_path=save_model_path,
                                 epoch_id=epoch_id,
                                 best_acc=acc)

    def export(self,
               save_model_path='models/',
               resume_model='models/panns_cnn10_LogMelSpectrogram/best_model/'):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        self.setup_model(input_size=self.input_size)
        # 加载预训练模型
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pt')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model_state_dict = torch.load(resume_model)
        self.model.load_state_dict(model_state_dict)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        self.model.eval()
        # 获取静态模型
        infer_model = self.model.state_dict()
        infer_model_path = os.path.join(
            save_model_path, f'{self.use_model}_{self.fea_mode}_{self.aug_mode}',
            'inference.pt')
        os.makedirs(os.path.dirname(infer_model_path), exist_ok=True)
        torch.save(infer_model, infer_model_path)
        logger.info("预测模型已保存：{}".format(infer_model_path))


if __name__ == "__main__":

    # 计时
    starttime = time.time()

    save_model_path = 'models/'

    fea_mode_all = ('LogMelSpectrogram', 'MelSpectrogram', 'MFCC', 'Spectrogram')
    aug_mode_all = ('origin', 'noise')
    use_model_all = ('panns_cnn6', 'panns_cnn10', 'panns_cnn14')
    input_size = {
        'LogMelSpectrogram': 128,
        'MelSpectrogram': 128,
        'MFCC': 13,
        'Spectrogram': 257
    }

    max_epoch = 100
    batch_size = 32

    fea_mode = fea_mode_all[0]
    aug_mode = aug_mode_all[0]
    use_model = use_model_all[0]
    learning_rate = 0.001

    my_train = Trainer(use_gpu=True,
                       fea_mode=fea_mode,
                       aug_mode=aug_mode,
                       batch_size=batch_size,
                       max_epoch=max_epoch,
                       use_model=use_model,
                       learning_rate=learning_rate,
                       input_size=input_size[fea_mode])

    my_train.train()

    model_path = os.path.join(save_model_path, f'{use_model}_{fea_mode}_{aug_mode}',
                              'best_model/')
    my_train.export(save_model_path=save_model_path, resume_model=model_path)

    # mode = "train"
    # filename = r"Dcase2020task1subset/"
    # aug_mode = None
    # fea_mode = ('LogMelSpectrogram', 'MelSpectrogram', 'MFCC', 'Spectrogram')
    # train_dataset = CustomDataset(data_list_path=filename,
    #                               aug_mode=aug_mode,
    #                               fea_mode=fea_mode[0],
    #                               mode=mode,
    #                               sample_rate=44100,
    #                               is_mixup=True)

    # for idx in range(train_dataset.__len__()):

    #     __, label = train_dataset.__getitem__(idx=idx)
    #     # print(label)
    # # print(train_data.__len__())

    endtime = time.time()
    all_time = round(endtime - starttime, 1)
    hour = int(all_time // 3600)
    minute = int((all_time - 3600 * hour) // 60)
    second = int(all_time - 3600 * hour - 60 * minute)
    print('总共的时间为:', hour, 'h', minute, 'min', second, 'sec')
    # a = train_data.__getitem__(idx=5)
