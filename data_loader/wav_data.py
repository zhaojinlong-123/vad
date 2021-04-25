import torch
import torchaudio
import os
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class WavDataset(Dataset):
    def __init__(self, dataset_path, bs, clip_length, val):
        super().__init__()
        self.dataset_path = dataset_path
        self.bs = bs
        self.clip_length = clip_length
        self.val = val


