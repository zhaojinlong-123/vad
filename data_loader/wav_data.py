import torch
import torchaudio
import os
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import csv
import math


class WavDataset(Dataset):
    def __init__(self, dataset_path, bs, clip_length, val):
        super().__init__()
        self.dataset_path = dataset_path
        self.bs = bs  # batch_size
        self.clip_length = clip_length  # 单位：毫秒
        self.val = val

        # 加载 CSV 标签文件
        labels_path = dataset_path + "_labels.csv"
        self.labels_dict = load_labels(labels_path, clip_length)

        self.len = 0
        self.labels_list = []
        self.data_list = []

        for k in self.labels_dict.keys():
            l = self.labels_dict[k]
            labels_cnt = l.__len__()
            self.len += labels_cnt
            self.labels_list.append(torch.Tensor(l))
            datafile_path = dataset_path + "/" + k + ".wav"
            data = load_data(datafile_path, clip_length)
            self.data_list.append(data[:, 0:labels_cnt])

        self.labels = torch.cat(self.labels_list)
        self.data = torch.cat(self.data_list, dim=1)

    def __len__(self):
        return int(math.ceil(self.len / self.bs))

    def __getitem__(self, index):
        start = index * self.bs
        end = min((index + 1) * self.bs, self.len)

        return self.data[:, start:end].T, self.labels[start:end]


def load_labels(labels_path, clip_length):
    """
    从数据文件中加载标签。
    参数：
        labels_path：csv 文件的路径
        clip_length: 片段的长度（毫秒）
    返回：
        字典类型。每一行的键是文件名，值是一个列表，包括了每个片段的标签。
    """
    labels = {}

    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row["id"]
            if file_id not in labels:  # 新文件
                current_time = 0  # 单位：毫秒
                labels[file_id] = []
            # start_time = float(row["s"]) * 1000  # 根本没用
            end_time = float(row["e"]) * 1000

            while current_time < end_time - clip_length / 2:  # 标签至少要覆盖超过一半的时长
                labels[file_id].append(int(row["label_index"]))
                current_time += clip_length

    return labels


def load_data(data_path, clip_length, feature_type="MFCC"):
    """
    加载一个 wav 文件的数据。
    参数：
        data_path：文件路径
        clip_length: 片段的长度（毫秒）
    返回：
        一个列表，由若干个一维张量构成。每个张量代表一个片段的特征。
    """
    data = []
    waveform, sample_rate = torchaudio.load(data_path)

    clip_size = int(clip_length * sample_rate / 1000)

    # 归一化
    mean = waveform.mean()
    std = waveform.std()
    waveform = (waveform - mean) / std

    assert feature_type in ["MFCC"]

    if feature_type == "MFCC":
        melkwargs = {
            "n_fft": clip_size,
            "hop_length": clip_size,
            "n_mels": 64
        }
        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=40, melkwargs=melkwargs)
        data = mfcc.forward(waveform)
        data = torch.squeeze(data, dim=0)

    return data


if __name__ == "__main__":
    dataset = WavDataset("../competition_dataset/train", 1, 10, False)
    dataset_len = dataset.__len__()
    print("dataset size:", dataset_len)

    for i in range(5):
        print(dataset.__getitem__(i))

    # 测试读取整个数据集，需要花几分钟时间
    '''
    for i in range(dataset_len):
        item = dataset.__getitem__(i)
    '''