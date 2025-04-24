import argparse
import os
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
# import .clip as clip
from .clip import clip
from .models.simvp_model import SimVP_Model
from .resnet import resnet34
from .utils import shift_dim


class VideoSimVP(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load VQ-VAE and set all parameters to no grad
        from .download import load_vqvae
        from .vqvae import VQVAE

        if not os.path.exists(args.vqvae):
            self.vqvae = load_vqvae(args.vqvae)
        else:
            self.vqvae = VQVAE.load_from_checkpoint(args.vqvae)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()
        self.criterion = nn.MSELoss()

        self.load_clip(args.clip)

        # Get the latent shape from VQ-VAE
        self.latent_shape = self.vqvae.latent_shape
        # Create SimVP model for latent prediction
        self.simvp = SimVP_Model(
            in_shape=(
                args.n_down_sample_cond_frames,  # not the cond frames, but the downsampled
                self.vqvae.embedding_dim,
                self.latent_shape[1],
                self.latent_shape[2],
            ),
            hid_S=args.hid_S,
            hid_T=args.hid_T,
            N_S=args.N_S,
            N_T=args.N_T,
            model_type=args.model_type,
            mlp_ratio=args.mlp_ratio,
            drop=args.dropout,
            drop_path=args.drop_path,
        )

        # caches for faster processing
        self.frame_cond_cache = None
        self.labels_features_cache = None
        

        self.save_hyperparameters()

    def encode_labels(self, labels: list[str]):
        text_tokens = clip.tokenize(labels).to("cuda")

        # now just run:
        # with torch.no_grad():   # if you’re only doing inference
        text_features = self.clip.encode_text(text_tokens)
        return text_features

    def load_clip(self, path):
        path = Path(path)
        model, _preprocess = clip.load(path, device="cuda")
        self.clip = model
        # we don't need the visual part of the clip, we can release it 
        # self.clip.visual = None
        # self.clip.train()
        # # freeze vision if you like:
        # for p in self.clip.visual.parameters():
        #     p.requires_grad_(False)
        # # not in eval mode
        self.clip.eval() # don't finetune clip
        for p in self.clip.parameters():
            p.requires_grad_(False)

    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        x = batch["video"]
        labels = batch["label"]
        # from loguru import logger
        # this is a list, we need clip to turns it into tensor
        # ing_step:66 - label: ['This is an deposition process, with parameters: deposition_time: 160.0 s, pressure: 1200.0 mTorr, power: 1200.0 W, space: 850.0 mil, SiH4_flow: 900.0 sccm, NH3_flow: 750.0 sccm, N2O_flow: 750.0 sccm, H2_flow: 2000.0 sccm, N2_flow: 2380.0 sccm.', 'This is an deposition process, with parameters: deposition_time: 160.0 s, pressure: 1200.0 mTorr, power: 1200.0 W, space: 850.0 mil, SiH4_flow: 360.0 sccm, NH3_flow: 300.0 sccm, N2O_flow: 450.0 sccm, H2_flow: 2000.0 sccm, N2_flow: 2380.0 sccm.', 'This is an etching process, with parameters: pressure: 120.0 MTorr, power: 100.0 W, temperature: 775.0 K, voltage: 10.0 V.', 'This is an etching process, with parameters: pressure: 120.0 MTorr, power: 260.0 W, temperature: 600.0 K, voltage: 20.0 V.', 'This is an deposition process, with parameters: deposition_time: 160.0 s, pressure: 1200.0 mTorr, power: 1200.0 W, space: 850.0 mil, SiH4_flow: 450.0 sccm, NH3_flow: 300.0 sccm, N2O_flow: 300.0 sccm, H2_flow: 2000.0 sccm, N2_flow: 2380.0 sccm.', 'This is an deposition process, with parameters: deposition_time: 160.0 s, pressure: 1200.0 mTorr, power: 1200.0 W, space: 850.0 mil, SiH4_flow: 150.0 sccm, NH3_flow: 300.0 sccm, N2O_flow: 300.0 sccm, H2_flow: 2000.0 sccm, N2_flow: 2380.0 sccm.', 'This is an deposition process, with parameters: deposition_time: 160.0 s, pressure: 1200.0 mTorr, power: 1200.0 W, space: 850.0 mil, SiH4_flow: 360.0 sccm, NH3_flow: 300.0 sccm, N2O_flow: 300.0 sccm, H2_flow: 1000.0 sccm, N2_flow: 2380.0 sccm.', 'This is an deposition process, with parameters: deposition_time: 160.0 s, pressure: 1200.0 mTorr, power: 1200.0 W, space: 850.0 mil, SiH4_flow: 360.0 sccm, NH3_flow: 300.0 sccm, N2O_flow: 300.0 sccm, H2_flow: 1500.0 sccm, N2_flow: 2380.0 sccm.']
        labels_features = self.encode_labels(labels)
        
        # from loguru import logger
        # logger.debug(f"labels_features shape: {labels_features.shape}")
        # labels_features shape: torch.Size([8, 512])

        # self.args.n_cond_frames is the input size and the output size
        # no matter what the predicted size.
        # for this model, it only takes in the same size of the input and the ouput
        # torch.Size([2, 3, 8, 128, 128])

        batch_x = x[:, :, : self.args.n_cond_frames, :, :]
        batch_y = x[:, :, self.args.n_cond_frames :, :, :]
        with torch.no_grad():
            # torch.Size([2, 2, 32, 32]),  torch.Size([2, 256, 2, 32, 32])
            encoding_x, embedding_x = self.vqvae.encode(
                batch_x, include_embeddings=True
            )
            # # torch.Size([2, 2, 32, 32, 256])
            embedding_x = shift_dim(embedding_x, 1, -1)

            encoding_y, embedding_y = self.vqvae.encode(
                batch_y, include_embeddings=True
            )
            embedding_y = shift_dim(embedding_y, 1, -1)

        predicted_y = self.forward(embedding_x)
        predicted_y = predicted_y.permute(0, 1, 3, 4, 2)
        # from loguru import logger
        # logger.debug(f"predicted_y shape\n{predicted_y.shape}")
        # logger.debug(f"embedding_y shape\n{embedding_y.shape}")
        # 2025-04-15 05:55:42.846 | DEBUG    | videogpt.simvp:training_step:132 - predicted_y shape
        # torch.Size([8, 2, 256, 32, 32])
        # 2025-04-15 05:55:42.846 | DEBUG    | videogpt.simvp:training_step:133 - embedding_y shape
        # torch.Size([8, 2, 32, 32, 256])

        loss = self.criterion(predicted_y, embedding_y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    # for the forward, it takes in the self.args.n_cond_frames frames and predcited the
    # same size of the output.
    def forward(self, x):
        # torch.Size([2, 2, 32, 32, 256])
        # batch size, downsample sequence length,  downsample height, downsample width, embedding dim
        # but we can treat it as
        # batch size, sequence length,  height, weight, hidden dim. a more layer.

        #  simvp takes in this    B, T, C, H, W = x_raw.shape

        # append a new dimension in the downsampled tensor

        # maybe not use the simvp model's forward, just its middle

        # OK, for new, we just treats the hidden dim as the channel dim

        # 假设你的张量名为 x
        x = x.permute(0, 1, 4, 2, 3)
        simvp_out = self.simvp(x)
        return simvp_out

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True)

    def sample(self, n, batch=None):
        """Generate new video samples."""

        # We need a batch of conditioning frames
        assert batch is not None, "Batch must be provided for conditioning"
        x = batch["video"]
        batch_x = x[:, :, : self.args.n_cond_frames, :, :]
        batch_y = x[:, :, self.args.n_cond_frames :, :, :]
        with torch.no_grad():
            encoding_x, embedding_x = self.vqvae.encode(
                batch_x, include_embeddings=True
            )
            embedding_x = shift_dim(embedding_x, 1, -1)
            predicted_y = self.forward(embedding_x)
            predicted_y = predicted_y.permute(0, 1, 3, 4, 2)
            # here you should calculate the distance from the predicted_y in the
            # codebook and retrieved

            # Flatten predicted_y for distance calculation with codebook
            B, T, H, W, C = predicted_y.shape
            predicted_flat = predicted_y.reshape(-1, C)

            # Get the codebook
            codebook = self.vqvae.codebook.embeddings

            # Calculate distances using squared Euclidean distance
            distances = (
                (predicted_flat ** 2).sum(dim=1, keepdim=True)
                - 2 * predicted_flat @ codebook.t()
                + (codebook.t() ** 2).sum(dim=0, keepdim=True)
            )

            # Find nearest codebook entries
            encoding_indices = torch.argmin(distances, dim=1)

            # Reshape encoding indices back to expected format for decode
            encoding_indices = encoding_indices.view(B, T, H, W)

            # Decode requires indices in format expected by VQVAE
            samples = self.vqvae.decode(encoding_indices)
            samples = torch.clamp(samples, -0.5, 0.5) + 0.5

        return samples  # BCTHW in [0, 1]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert (
            hasattr(self.args, "max_steps") and self.args.max_steps is not None
        ), f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval="step", frequency=1)]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--vqvae",
            type=str,
            default="kinetics_stride4x4x4",
            help="path to vqvae ckpt, or model name to download pretrained",
        )
        parser.add_argument("--clip", type=str, required=True, 
                            help="path to openai clip model")

        parser.add_argument("--n_cond_frames", type=int, default=1)
        parser.add_argument(
            "--n_pred_frames",
            type=int,
            default=1,
            help="number of frames to predict (default: same as n_cond_frames)",
        )

        # SimVP hyperparameters
        parser.add_argument(
            "--n_down_sample_cond_frames",
            type=int,
            default=2,
            help="number of downsampled frames"
            "downsampled by vqvae, will be input into simvp",
        )
        parser.add_argument("--hid_S", type=int, default=64)
        parser.add_argument("--hid_T", type=int, default=512)
        parser.add_argument("--N_S", type=int, default=4)
        parser.add_argument("--N_T", type=int, default=8)
        parser.add_argument("--model_type", type=str, default="gSTA")
        parser.add_argument("--mlp_ratio", type=float, default=8.0)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--drop_path", type=float, default=0.1)

        return parser
