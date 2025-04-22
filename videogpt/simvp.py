import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

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

        self.save_hyperparameters()

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    # def compute_loss(self, batch):
    #     """Compute training or validation loss."""
    #     x = batch["video"]
    #
    #     # Forward pass
    #     pred_embeddings, target_encodings = self(x)
    #
    #     # Get prediction length
    #     if hasattr(self.args, "n_pred_frames") and self.args.n_pred_frames is not None:
    #         n_pred_frames = self.args.n_pred_frames
    #     else:
    #         n_pred_frames = self.args.n_cond_frames
    #
    #     # Compute reconstruction loss
    #     # Get the predicted frames (excluding conditioning frames)
    #     pred_frames = self.vqvae.decode(pred_embeddings.reshape(-1, *self.latent_shape))
    #
    #     # Reshape pred_frames to match video shape
    #     B = x.shape[0]
    #     pred_frames = pred_frames.reshape(B, -1, *x.shape[2:])
    #
    #     # Get the appropriate target frames based on prediction length
    #     if n_pred_frames <= self.args.n_cond_frames:
    #         target_frames = x[
    #             :, :, self.args.n_cond_frames : self.args.n_cond_frames + n_pred_frames
    #         ]
    #     else:
    #         # For autoregressive prediction, we need frames beyond conditioning
    #         target_frames = x[
    #             :, :, self.args.n_cond_frames : self.args.n_cond_frames + n_pred_frames
    #         ]
    #         # In case we don't have enough target frames in dataset, truncate prediction
    #         if target_frames.shape[2] < n_pred_frames:
    #             pred_frames = pred_frames[:, :, : target_frames.shape[2]]
    #
    #     # Flatten for MSE calculation
    #     pred_frames = pred_frames.reshape(-1, *pred_frames.shape[3:])
    #     target_frames = target_frames.reshape(-1, *target_frames.shape[3:])
    #
    #     recon_loss = F.mse_loss(pred_frames, target_frames)
    #
    #     return recon_loss

    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        x = batch["video"]
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

        # Get the latent embeddings from VQ-VAE
        # with torch.no_grad():
        #     # Reshape for VQ-VAE encoding
        #     B, C, T, H, W = x.shape
        #     x_flat = x.reshape(B * T, C, H, W)
        #
        # # Get encodings and embeddings from VQ-VAE
        # encodings, embeddings = self.vqvae.encode(x_flat, include_embeddings=True)
        #
        # # Reshape back to batch form
        # embeddings = embeddings.reshape(
        #     B, T, -1, self.latent_shape[1], self.latent_shape[2]
        # )
        # encodings = encodings.reshape(B, T, *self.latent_shape)

        #
        # # Generate predictions in chunks
        # for _ in range(d):
        #     cur_pred = self.simvp(cur_frames)
        #     pred_embeddings.append(cur_pred)
        #     cur_frames = cur_pred  # Use predictions as next input
        #
        # # Handle remaining frames if needed
        # if m > 0:
        #     cur_pred = self.simvp(cur_frames)
        #     pred_embeddings.append(cur_pred[:, :m])

        # # Concatenate all predictions
        # simvp_out = torch.cat(pred_embeddings, dim=1)

        # Return the predicted embeddings and the target encodings

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
                (predicted_flat**2).sum(dim=1, keepdim=True)
                - 2 * predicted_flat @ codebook.t() 
                + (codebook.t()**2).sum(dim=0, keepdim=True)
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
