from data_loader import wav_data
import model.models
from test import LitModel
import torchaudio
from config import Config
from config import DataConfig
import os
import torch
import pytorch_lightning as pl
from glob import glob
from tqdm import tqdm


def change_labels_to_csv(uid, labels, clip_length):
    curr_time = 0
    status = 0  # Non Speech

    csv_list = []

    start_time = end_time = 0

    for i in range(len(labels)):
        curr_time = i * clip_length / 1000

        if status == 0 and labels[i] == 1:
            start_time = curr_time
            status = 1

        elif status == 1 and labels[i] == 0:
            end_time = curr_time
            status = 0
            csv_list.append((uid, start_time, end_time))

    return csv_list


if __name__ == "__main__":
    dataset_path = "competition_dataset/test"
    dataconfig = DataConfig()
    file_list = os.listdir(dataset_path)
    clip_length = dataconfig.clip_length

    cfg = Config()
    #checkpoint = "saved_models/original_0.0005_softmax_cliplength50/epoch=0106-val_loss=1.231-f1=0.914-acc=0.914-precision=0.914-recall=0.914.ckpt"
    checkpoint = "saved_models/original_0.0005_softmax_cliplength50/epoch=0070-val_loss=1.233-f1=0.915-acc=0.915-precision=0.915-recall=0.915.ckpt"
    model = LitModel(cfg)
    model = model.load_from_checkpoint(checkpoint, cfg=cfg, strict=False)
    model.train(False)

    output_file_name = "test_pre.csv"

    i = 1
    labels_list = []

    for f in file_list:
        print("Prossing %s (%d/%d)" % (f, i, len(file_list)))
        uid = f.replace(".wav", "")

        data_path = dataset_path + "/" + f
        data = wav_data.load_data(data_path, clip_length)
        data = data.T
        data = data.unsqueeze(0)
        #print(data.shape)

        y_p = model(data)
        y_p = torch.softmax(y_p, dim=2)
        predict_result = torch.argmax(y_p, dim=2)
        y_pb = (predict_result != 3).int()
        y_pb = torch.squeeze(y_pb, dim=0)

        l = change_labels_to_csv(uid, y_pb, clip_length)
        labels_list.extend(l)
        i += 1


    with open(output_file_name, "w") as f:
        f.write("id,s,e\n")
        for entry in labels_list:
            f.write("%s,%f,%f\n" % (entry[0], entry[1], entry[2]))



