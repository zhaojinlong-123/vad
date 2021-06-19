import os


class DataConfig:
    def __init__(self):
        self.train_dataset_path = "competition_dataset/train"
        self.val_dataset_path = "competition_dataset/val"
        self.test_dataset_path = "competition_dataset/test"
        self.bs = 150000
        self.clip_length = 20
        self.NON_SPEECH_LABEL = 3


class TrainConfig:
    def __init__(self):
        self.debug = False
        self.lr = 5e-5
        # self.lr = 1e-5
        self.n_val = 1


class Config:
    def __init__(self):
        self.data_cfg = DataConfig()
        self.train_cfg = TrainConfig()
