#!/usr/bin/env python3

import os
import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, random_split

from lib.data import MaskDataset, ReduceSize, PILToTensor, PilConvertToRGB
from lib.model import MobileNetV2Unet


class DataModule(pl.LightningDataModule):
    """DataSet module"""

    def __init__(
        self,
        data_dir: str = "./dataset",
        batch_size: int = 32,
        valid_percent: float = 0.1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.valid_percent = valid_percent
        self.train_data_dir = os.path.join(data_dir, "train")
        self.num_workers = 8
    
    def setup(self, stage: Optional[str] = None):
        ds = self._create_train_dataset()
        tlen = int(len(ds) * (1 - self.valid_percent))
        vlen = len(ds) - tlen
        self.ds_train, self.ds_valid = random_split(ds, [tlen, vlen])
    
    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def _create_train_dataset(self):
        ds = MaskDataset(
            root_dir=self.train_data_dir,
            transform=Compose([
                PilConvertToRGB(),  # L to RGB
                ToTensor(),  # C, H, W
            ]),
            transform_mask=Compose([
                ReduceSize(2),
                PILToTensor(),  # H, W
            ]),
        )
        return ds


class MobileUnetModule(pl.LightningModule):
    """Module for train modele"""

    def __init__(self, backbone: nn.Module, learning_rate=1e-3):
        """
        Args:
            backbone - model for train
        """
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone

    def forward(self, x):
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("valid_loss", loss)
        return loss
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False
        )
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument(
            "--classes",
            type=int,
            default=1,
            help="The count of classses for predict"
        )
        return parser


def tuple_type(value: str) -> Tuple[int, int]:
    return tuple(map(int, value.split(",")))


def cli_main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument(
        "--input-size",
        type=tuple_type,
        default=(224, 224),
        help="This is the size of image which will be taken to input (w, h)"
    )
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MobileUnetModule.add_model_specific_args(parser)
    args = parser.parse_args()

    # Data
    data_module = DataModule("./dataset_ph", args.batch_size)

    # Model
    model = MobileNetV2Unet(classes=args.classes)
    module = MobileUnetModule(backbone=model, learning_rate=args.learning_rate)

    # Training
    checkpoint_end = pl.callbacks.ModelCheckpoint()
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_end], num_sanity_val_steps=0
    )
    trainer.fit(module, data_module)

    # To onnx
    print("Convert to onnx")
    # batch_size, channel, h, w
    input_sampe = torch.randn((1, 3, args.input_size[1], args.input_size[0]))
    module.to_onnx(
        os.path.join(checkpoint_end.dirpath, "..", "model_last.onnx"),
        input_sampe,
        export_params=True,
    )

    # Testing


if __name__ == "__main__":
    cli_main()
