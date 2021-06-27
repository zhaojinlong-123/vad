import os


class DataConfig:
    def __init__(self):
        self.train_dataset_path = "competition_dataset/train"
        self.val_dataset_path = "competition_dataset/test"
        self.test_dataset_path = "competition_dataset/test"
        self.bs = 50000#50000
        self.clip_length = 100
        self.NON_SPEECH_LABEL = 3
        self.window_shift = 50


class TrainConfig:
    def __init__(self):
        self.debug = False
        self.lr = 5e-4
        # self.lr = 1e-5
        self.n_val = 1


class Config:
    def __init__(self):
        self.data_cfg = DataConfig()
        self.train_cfg = TrainConfig()
