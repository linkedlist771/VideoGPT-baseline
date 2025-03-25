import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from videogpt import VQVAE, VideoData
import datetime


def main():
    pl.seed_everything(1234)

    # Get current date for checkpoint directory
    current_date = datetime.datetime.now()
    month_day = f"{current_date.month:02d}_{current_date.day:02d}"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/wilson/data/datasets/bair.hdf5')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    
    # Add VQVAE specific arguments
    parser.add_argument('--embedding_dim', type=int, default=256,
                      help='Dimension of embeddings')
    parser.add_argument('--n_codes', type=int, default=2048,
                      help='Number of codes in codebook')
    parser.add_argument('--n_hiddens', type=int, default=240,
                      help='Number of hidden units')
    parser.add_argument('--n_res_layers', type=int, default=4,
                      help='Number of residual layers')
    parser.add_argument('--downsample', nargs='+', type=int, default=[4, 4, 4],
                      help='Downsampling factors')
    parser.add_argument('--sync_batchnorm', action='store_true',
                      help='Use synchronized batch normalization')
    parser.add_argument('--save_dir', type=str, default=f'checkpoints/vqvae/{month_day}',
                      help='Directory to save VQVAE checkpoints')
    
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()
    model = VQVAE(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(
        dirpath=args.save_dir,
        filename='vqvae_{epoch:02d}',
        monitor='val/recon_loss',
        mode='min',
        save_last=True,
        save_top_k=2
    ))


    

    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'accelerator': 'gpu' if args.gpus > 0 else 'cpu',
        "devices": 1,
        'callbacks': callbacks,
        'gradient_clip_val': args.gradient_clip_val,
    }

    if args.gpus > 1:
        trainer_kwargs.update({
            'strategy': 'ddp',
        })

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, data)
    
    trainer.save_checkpoint(f"{args.save_dir}/vqvae_final.ckpt")


if __name__ == '__main__':
    main()

