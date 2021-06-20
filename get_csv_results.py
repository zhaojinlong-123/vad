from data_loader import wav_data
import model.models
from test import LitModel
import torchaudio
from config import Config
from config import DataConfig
import os

from glob import glob
from tqdm import tqdm





if __name__ == "__main__":
    dataset_path = "competition_dataset/val"
    dataconfig = DataConfig()
    file_list = os.listdir(dataset_path)
    clip_length = dataconfig.clip_length

    cfg = Config()
    checkpoint = "saved_models/mfcc_5e-5/epoch=0173-val_loss=0.963-f1=0.902-acc=0.902-precision=0.902-recall=0.902.ckpt"
    model = LitModel(cfg)
    model = model.load_from_checkpoint(checkpoint, cfg=cfg, strict=False)
    model.train(False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    i = 1
    for f in file_list:
        print("Prossing %s (%d/%d)" % (f, i, len(file_list)))
        uid = f.replace(".wav", "")

        data_path = dataset_path + "/" + f
        data = wav_data.load_data(data_path, clip_length)
        data = data.T
        data = data.unsqueeze(0)
        #print(data.shape)

        #label = model(data)
        #print(label)


        i += 1



