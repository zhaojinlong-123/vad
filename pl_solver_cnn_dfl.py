from model.models import CRNN
import torch
import pytorch_lightning as pl
from data_loader.wav_data import WavDataset
from config import Config
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from torchmetrics import F1
import torchaudio
import shutil
import numpy as np


class LitModel(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.train_cfg.lr
        self.save_hyperparameters()
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = CRNN(160, 4)
        self.f1 = pl.metrics.F1(num_classes=2)
        self.acc = pl.metrics.Accuracy()
        self.pre = pl.metrics.Precision()
        self.recall = pl.metrics.Recall()
        # self.test_metrics = torchmetrics.MetricCollection([Accuracy(), Precision(), Recall()])

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        # opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        monitor = 'val_loss'
        scheduler = ReduceLROnPlateau(opt, "min", factor=0.8, min_lr=self.lr/10, patience=10)
        schedulers = [
            {
                'scheduler': scheduler,
                'monitor': monitor,
                'interval': 'epoch',
                'frequency': self.cfg.train_cfg.n_val
            }
        ]
        self.opt = opt
        return [opt], schedulers

    def predict_batch(self, batch):
        #x = batch['features']
        #y = batch['labels']
        x, y = batch
        x = repeat(x, "bs dim -> bs cliplength dim",cliplength=4)
        y = repeat(y, "bs-> bs cliplength",cliplength=4)
        #x = rearrange(x, "bs bucketsize cliplength dim -> (bs bucketsize) cliplength dim")
        #y = rearrange(y, "bs bucketsize cliplength-> (bs bucketsize) cliplength")
        y_predict = self(x)
        return x, y_predict, y

    def cal_loss_and_log_info(self, y_p, y, val):
        if val:
            prefix = "val_"
        else:
            prefix = ""
        y_p = rearrange(y_p, "bs time dim -> bs dim time")
        loss = self.loss(y_p, y)
        self.log(F"{prefix}loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        x, y_p, y = self.predict_batch(batch)
        loss = self.cal_loss_and_log_info(y_p, y, val=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_p, y = self.predict_batch(batch)
        _ = self.cal_loss_and_log_info(y_p, y, val=True)

        non_speech_label = self.cfg.data_cfg.NON_SPEECH_LABEL
        y_b = (~(y == non_speech_label)).long()
        y_b = rearrange(y_b, "bs time-> (bs time)")

        y_p = torch.softmax(y_p, dim=2)
        prob_non_speech = y_p[:, :, non_speech_label]
        prob_speech = 1 - prob_non_speech
        y_pb = torch.stack([prob_non_speech, prob_speech], dim=2)

        y_pb = rearrange(y_pb, "bs time dim -> (bs time) dim")
        # self.val_metrics(y_pb, y_b)
        self.f1(y_pb, y_b)
        self.acc(y_pb, y_b)
        self.pre(y_pb, y_b)
        self.recall(y_pb, y_b)
        self.log("f1", self.f1)
        self.log("acc", self.acc)
        self.log("precision", self.pre)
        self.log("recall", self.recall)

    def test_step(self, batch, batch_idx):
        pass

    def train_dataloader(self):
        train_dataset = WavDataset(self.cfg.data_cfg.train_dataset_path,
                                   self.cfg.data_cfg.bs,
                                   clip_length=self.cfg.data_cfg.clip_length,
                                   val=False)
        if self.cfg.train_cfg.debug:
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
        else:
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
            # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=56)
        return train_loader

    def val_dataloader(self):
        val_dataset = WavDataset(self.cfg.data_cfg.val_dataset_path,
                                 self.cfg.data_cfg.bs,
                                 clip_length=self.cfg.data_cfg.clip_length,
                                 val=True)
        if self.cfg.train_cfg.debug:
            val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=False)
        else:
            val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=False)
            # val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=False, num_workers=56)
        return val_loader

    def test_dataloader(self):
        val_dataset = WavDataset(self.cfg.data_cfg.val_dataset_path,
                                 self.cfg.data_cfg.bs,
                                 clip_length=self.cfg.data_cfg.clip_length,
                                 val=True)
        if self.cfg.train_cfg.debug:
            val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=False)
        else:
            val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=False)
            # val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=False, num_workers=56)
        return val_loader


if __name__ == "__main__":
    cfg = Config()
    # n_gpus = 2
    n_gpus = "5"
    # model = LitModel(lr=5e-4)
    # debug = True
    debug = False
    # test = True
    test = False

    description = "original"
    cfg.train_cfg.debug = debug
    model = LitModel(cfg)
    # checkpoint = "./saved_models/dfl_speech_dfl_noise_None_0.0002/epoch=0149-pesq=3.446-stoi=0.940.ckpt"
    checkpoint = None
    if checkpoint is not None:
        checkpoint_path = os.path.dirname(checkpoint)
        model = model.load_from_checkpoint(checkpoint, cfg=cfg, strict=False)
    version = F"{description}_{cfg.train_cfg.lr}"

    checkpoint_callback = ModelCheckpoint(save_top_k=5, mode="min", monitor="val_loss", verbose=True,
                                          dirpath=F"saved_models/{version}",
                                          # filename="{epoch:04d}-{f1:.3f}-{acc:.3f}-{precision:.3f}-{recall:.3f}")
                                          filename="{epoch:04d}-{val_loss:.3f}-{f1:.3f}")
    lr_logger = LearningRateMonitor(logging_interval='epoch')
    if debug:
        trainer = pl.Trainer(gpus=1,
                             # checkpoint_callback=checkpoint_callback,
                             # callbacks=[lr_logger],
                             # limit_train_batches=20,
                             # limit_val_batches=4,
                             # check_val_every_n_epoch=1,
                             precision=16,
                             )
        # trainer.tune(model)
    else:
        tb_logger = TensorBoardLogger("tb_results", name=version)
        trainer = pl.Trainer(gpus=n_gpus,
                             checkpoint_callback=checkpoint_callback,
                             callbacks=[lr_logger],
                             accumulate_grad_batches=2,
                             accelerator="ddp",
                             check_val_every_n_epoch=cfg.train_cfg.n_val,
                             precision=16,
                             max_epochs=2000,
                             logger=[tb_logger]
                             )
    if not test:
        trainer.fit(model)
    else:
        # os.makedirs(F"{checkpoint_path}/{OUTPUT}", exist_ok=True)
        # for f in os.listdir(F"{checkpoint_path}/{OUTPUT}"):
        #     os.remove(F"{checkpoint_path}/{OUTPUT}/{f}")
        # for f in os.listdir(F"{OUTPUT}"):
        #     os.remove(F"{OUTPUT}/{f}")
        trainer.test(model)
