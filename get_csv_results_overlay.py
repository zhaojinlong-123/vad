import numpy as np

from data_loader import wav_data_overlay
#import model.models
#import model.BLSTMRNN
from pl_solver_cnn_dfl_2 import LitModel
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
    
def change_4_labels_to_csv(uid, labels, clip_length):
    curr_time = 0
    labels = np.squeeze(labels, 0)
    status = labels[0]

    csv_list = []
    
    start_time = end_time = 0
    for i in range(len(labels)):
        curr_time = i * clip_length / 1000

        if status != labels[i]:
            end_time = curr_time
            csv_list.append((uid, start_time, end_time, status))
            status = labels[i]
            start_time = curr_time
    return csv_list

if __name__ == "__main__":
    dataset_path = "competition_dataset/test"
    dataconfig = DataConfig()
    file_list = os.listdir(dataset_path)
    clip_length = dataconfig.clip_length

    is_two_classes = True  # 是否为二分类

    cfg = Config()
    checkpoint = "saved_models/original_0.0005_softmax_cliplength50_BLSTM_L4T_F64_B5/epoch=0008-val_loss=1.118-f1=0.869.ckpt"
    #checkpoint = "saved_models/original_0.0005_softmax_cliplength50_BLSTMRNN_L4T_B5/epoch=0101-val_loss=0.984-f1=0.906-acc=0.906-precision=0.906-recall=0.906.ckpt"
    
    #checkpoint = "saved_models/original_0.0005_softmax_cliplength50_BLSTMRNN_L4_B5/epoch=0096-val_loss=0.954-f1=0.921-acc=0.921-precision=0.921-recall=0.921.ckpt"
    #checkpoint = "saved_models/original_0.0005_softmax_cliplength50_BLSTMRNN_B5/epoch=0044-val_loss=0.939-f1=0.917-acc=0.917-precision=0.917-recall=0.917.ckpt"
    #checkpoint = "saved_models/original_0.0005_softmax_cliplength50_testdataset/epoch=0186-val_loss=0.941-f1=0.917-acc=0.917-precision=0.917-recall=0.917.ckpt"
    #checkpoint = "saved_models/original_0.0005_softmax_cliplength50_CRNN_B5/epoch=0047-val_loss=0.949-f1=0.917-acc=0.917-precision=0.917-recall=0.917.ckpt"
    #checkpoint = "saved_models/original_0.0005_softmax_cliplength50/epoch=0070-val_loss=1.233-f1=0.915-acc=0.915-precision=0.915-recall=0.915.ckpt"
    #checkpoint = "model_mfcc/epoch=0061-val_loss=0.942-f1=0.912-acc=0.912-precision=0.912-recall=0.912.ckpt"
    model = LitModel(cfg)
    model = model.load_from_checkpoint(checkpoint, cfg=cfg, strict=False)
    model.train(False)

    output_file_name = "test_pre_7.csv"

    i = 1
    labels_list = []

    if is_two_classes:
        for f in file_list:
            print("Prossing %s (%d/%d)" % (f, i, len(file_list)))
            uid = f.replace(".wav", "")

            data_path = dataset_path + "/" + f
            data = wav_data_old.load_data(data_path, clip_length)
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

    else:
        for f in file_list:
            print("Prossing %s (%d/%d)" % (f, i, len(file_list)))
            uid = f.replace(".wav", "")

            data_path = dataset_path + "/" + f
            data = wav_data_old.load_data(data_path, clip_length)
            data = data.T
            data = data.unsqueeze(0)

            y_p = model(data)
            y_p = torch.softmax(y_p, dim=2)
            predict_result = torch.argmax(y_p, dim=2)

            l = change_4_labels_to_csv(uid, predict_result, clip_length)
            labels_list.extend(l)
            i += 1


        with open(output_file_name, "w") as f:
            f.write("id,s,e,label_index\n")
            for entry in labels_list:
                f.write("%s,%f,%f,%d\n" % (entry[0], entry[1], entry[2], entry[3]))



