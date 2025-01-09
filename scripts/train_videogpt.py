import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from videogpt import VideoGPT, VideoData
import os


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    # Add basic arguments
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)

    # Add training arguments that were previously in Trainer.add_argparse_args
    parser.add_argument('--accelerator', type=str, default='gpu')
    # parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)  # for backward compatibility
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)

    # Add VideoGPT model specific arguments
    parser.add_argument('--vqvae', type=str, default='kinetics_stride4x4x4')
    parser.add_argument('--n_cond_frames', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=576)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attn_type', type=str, default='full')
    parser.add_argument('--attn_dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--class_cond', action='store_true')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=0)

    # Mixed precision training arguments
    parser.add_argument('--amp_level', type=str, default='O1')

    # Add save directory argument
    parser.add_argument('--save_dir', type=str, default='checkpoints/videogpt',
                        help='Directory to save VideoGPT checkpoints')

    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    args.class_cond_dim = data.n_classes if args.class_cond else None
    model = VideoGPT(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(
        dirpath=args.save_dir,
        filename='videogpt_{epoch:02d}',
        monitor='val/loss',
        mode='min',
        save_top_k=-1
    ))

    kwargs = dict()
    # if args.devices > 1 or args.gpus > 1:
    #     kwargs = dict(strategy='ddp')
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    # print(cuda_visible_devices)
    # if cuda_visible_devices:
    #     devices = [int(x) for x in cuda_visible_devices.split(",")]
    # else:
    #     devices = 0
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        # devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        devices=1,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        max_steps=args.max_steps,
        **kwargs
    )

    trainer.fit(model, data)


if __name__ == '__main__':
    main()
