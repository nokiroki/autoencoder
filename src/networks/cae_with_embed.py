__all__ = ["Conv1dAutoEncoder"]

import time

import pytorch_lightning as pl
import torch
import torch.nn as nn


def init_weights(m):
    """
    Simple weight initialization
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        m.bias.data.fill_(0.01)


class Conv1dAutoEncoder(pl.LightningModule):
    """
    Main block of convolutional event clustering
    encoder-decoder architecture allows to create representation of Cohortney features
    """

    def __init__(
        self,
        in_channels: int,
        n_latent_features: int,
        vocab_size: int = 204,
        embed_dim: int = 20,
        dropout: float = .4
    ):
        super().__init__()
        self.out = n_latent_features

        self.small_group_embed = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels + embed_dim - 1, out_channels=512, kernel_size=3),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=32, out_channels=self.out, kernel_size=3),
        )
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.out, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(in_channels=512, out_channels=in_channels + embed_dim - 1, kernel_size=3),
            nn.Conv1d(in_channels=in_channels + embed_dim - 1, out_channels=in_channels, kernel_size=1)
        )
        self.decoder.apply(init_weights)

        self.train_index = 0
        self.val_index = 0
        self.final_labels = None
        self.time_start = time.time()

    def forward(self, x):

        x, c = torch.split(x, [3, 1], 2)

        c = self.small_group_embed(c)
        x = torch.cat((x, c), 2)

        latent = self.encoder(x)
        return latent

    def predict_step(self, x):
        latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x)
        return {'latent': latent, 'loss': loss}

    def training_step(self, batch, batch_idx):
        x = batch
        # print(x.shape)
        # print(gts.shape)
        latent = self(x)
        # print(latent.shape)
        loss = torch.nn.MSELoss()(self.decoder(latent), x)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.log("train_time", time.time() - self.time_start, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x = batch
        latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx: int):
        x = batch
        latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "latent": latent}

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

        def adjust_lr(epoch):
            if epoch < 100:
                return 0.003
            if 100 <= epoch < 120:
                return 0.0003
            if 120 <= epoch < 150:
                return 0.000003
            if 150 <= epoch < 200:
                return 0.0000003
            else:
                return 0.00000003

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=adjust_lr
            ),
            "name": "lr schedule",
        }
        return [optimizer], [lr_scheduler]
