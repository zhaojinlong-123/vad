import torch
import torchaudio
import os
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import csv


class WavDataset(Dataset):
    def __init__(self, dataset_path, bs, clip_length, val):
        super().__init__()
        self.dataset_path = dataset_path
        self.bs = bs  # batch size
        self.clip_length = clip_length  # 单位：毫秒
        self.val = val

        # 加载 CSV 标签文件
        labels_path = dataset_path + "_labels.csv"
        self.labels = load_labels(labels_path, clip_length)

        # 统计数量
        self.len = 0
        for l in self.labels.values():
            self.len += l.__len__()

        # 不能一次性载入全部数据，否则内存会爆炸：
        """
        # 加载波形数据文件
        for id in self.labels.keys():
            datafile_path = dataset_path + "/" + id + ".wav"
            data = load_data(datafile_path, clip_length)
        """
        # 缓存
        self.index_cache = (0, list(self.labels.values())[0].__len__())
        self.id_cache = list(self.labels.keys())[0]
        self.data_cache = load_data(self.dataset_path + "/" + self.id_cache + ".wav", self.clip_length)

        # 有些数据的标签居然少了几十毫秒，造成 label 和 data 的长度不一致……
        # 暂且先把多余的 data 截断吧。
        if self.data_cache.__len__() > list(self.labels.values())[0].__len__():
            self.data_cache = self.data_cache[:list(self.labels.values())[0].__len__()]

    def __getoneitem__(self, index):
        # 如果 index 不在缓存范围内，则需要刷新缓存
        if not (self.index_cache[0] <= index < self.index_cache[1]):
            head = 0
            end = 0
            i = 0
            for l in self.labels.values():
                end = head + l.__len__()
                if head <= index < end:
                    self.index_cache = (head, end)
                    self.id_cache = list(self.labels.keys())[i]
                    self.data_cache = load_data(self.dataset_path + "/" + self.id_cache + ".wav", self.clip_length)

                    # 同上。长度不一问题。
                    if self.data_cache.__len__() > l.__len__():
                        self.data_cache = self.data_cache[:l.__len__()]
                i += 1
                head = end

        # index 在缓存范围内，直接从缓存中提取
        data_idx = index - self.index_cache[0]  # 将实际的 index 转换成 data 中对应的下标
        data = self.data_cache[data_idx]
        label = self.labels[self.id_cache][data_idx]
        return data, label

    def __getitem__(self, index):
        idx_start = index * self.bs
        idx_end = (index + 1) * self.bs

        datas = []
        labels = []

        for idx in range(idx_start, idx_end):
            data_tmp, label_tmp = self.__getoneitem__(idx)
            datas.append(data_tmp)
            labels.append(label_tmp)

        return torch.stack(datas, 0), torch.tensor(labels)


    def __len__(self):
        t = self.len // self.bs
        return t


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


def load_data(data_path, clip_length):
    """
    加载一个 wav 文件的数据。
    参数：
        data_path：文件路径
        clip_length: 片段的长度（毫秒）
    返回：
        一个列表，由若干个一维张量构成。每个张量代表一个片段的特征。
        目前采用的特征是频谱幅值。
    """
    data = []

    raw_wav, sample_rate = torchaudio.load(data_path)
    raw_wav = raw_wav.squeeze(0)
    clip_size = int(clip_length / 1000 * sample_rate)

    head = 0
    while head < raw_wav.size()[0] - clip_size / 2:
        clip = raw_wav[head:head + clip_size]
        clip_fft = torch.abs(torch.fft.fft(clip))  # 离散傅里叶变换。人耳对相位不敏感，因此只取幅值。
        data.append(clip_fft)
        head += clip_size

    return data
    
 
if __name__ == "__main__":
    dataset = WavDataset("../competition_dataset/train", 10, 10, False)
    dataset_len = dataset.__len__()
    print("dataset size:", dataset_len)

    for i in range(1):
        print(dataset.__getitem__(i))

    # 测试读取整个数据集，需要花几分钟时间
    '''
    for i in range(dataset_len):
        item = dataset.__getitem__(i)
    '''
 