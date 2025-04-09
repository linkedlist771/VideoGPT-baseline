import argparse
import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from videogpt import VideoSimVP, VideoData


def main():
    pl.seed_everything(1234)

    # Get current date for checkpoint directory
    current_date = datetime.datetime.now()
    month_day = f"{current_date.month:02d}_{current_date.day:02d}"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/home/wilson/data/datasets/bair.hdf5"
    )
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=f"checkpoints/simvp/{month_day}",
        help="Directory to save SimVP checkpoints",
    )

    # Add VideoSimVP specific arguments
    parser = VideoSimVP.add_model_specific_args(parser)

    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.val_dataloader()
    model = VideoSimVP(args)

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="simvp_{epoch:02d}",
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=2,
        )
    )

    trainer_kwargs = {
        "max_steps": args.max_steps,
        "accelerator": "gpu" if args.gpus > 0 else "cpu",
        "devices": args.gpus if args.gpus > 0 else None,
        "callbacks": callbacks,
        "gradient_clip_val": args.gradient_clip_val,
        "precision": args.precision,
        "val_check_interval": 0.1,  # validate every 10% of training steps
        "log_every_n_steps": 10,
    }

    if args.gpus > 1:
        trainer_kwargs.update(
            {"strategy": "ddp",}
        )

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, data)

    trainer.save_checkpoint(f"{args.save_dir}/simvp_final.ckpt")

    # Generate and save sample predictions
    import os
    import torch
    from torchvision.utils import save_image

    os.makedirs(f"{args.save_dir}/samples", exist_ok=True)
    model.eval()

    with torch.no_grad():
        batch = next(iter(data.val_dataloader()))
        samples = model.sample(4, batch)

        # Reshape to sequence of frames
        B, T, C, H, W = samples.shape
        samples = samples.reshape(B * T, C, H, W)

        # Save as grid
        save_image(samples, f"{args.save_dir}/samples/simvp_samples.png", nrow=T)
        print(f"Saved samples to {args.save_dir}/samples/simvp_samples.png")


if __name__ == "__main__":
    main()
