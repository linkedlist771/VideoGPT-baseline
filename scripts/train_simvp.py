import argparse
import datetime
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from videogpt import VideoData, VideoSimVP


def main():
    pl.seed_everything(1234)
    current_date = datetime.datetime.now()

    month_day = f"{current_date.month:02d}_{current_date.day:02d}"

    parser = argparse.ArgumentParser()
    # Add basic arguments
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    # Add training arguments that were previously in Trainer.add_argparse_args
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--gpus", type=int, default=1)  # for backward compatibility
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    # SimVP hyperparameters
    parser.add_argument(
        "--vqvae",
        type=str,
        default="kinetics_stride4x4x4",
        help="path to vqvae ckpt, or model name to download pretrained",
    )
    parser.add_argument(
        "--clip", type=str, required=True, help="path to openai clip model"
    )
    parser.add_argument("--n_cond_frames", type=int, default=1)
    parser.add_argument(
        "--n_pred_frames",
        type=int,
        default=1,
        help="number of frames to predict (default: same as n_cond_frames)",
    )
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

    # Clip attention projector params
    parser.add_argument("--output_channels", type=int, required=True)
    parser.add_argument("--output_h", type=int, required=True)
    parser.add_argument("--output_w", type=int, required=True)

    # Add save directory argument
    parser.add_argument(
        "--save_dir",
        type=str,
        default=f"checkpoints/videosimvp/{month_day}",
        help="Directory to save VideoSimVP checkpoints",
    )

    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    model = VideoSimVP(args)

    callbacks = []
    # 只保存最后三个
    callbacks.append(
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="videosimvp_{epoch:02d}",
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=2,
        )
    )

    kwargs = dict()
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        precision=args.precision,
        devices=1,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        max_steps=args.max_steps,
        **kwargs,
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
