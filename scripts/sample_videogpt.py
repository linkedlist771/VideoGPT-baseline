import os
import argparse
import torch
from pathlib import Path
from videogpt import VideoData, VideoGPT, load_videogpt
from videogpt.utils import save_video_grid
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='ucf101_uncond_gpt')
parser.add_argument('--n', type=int, default=8)
parser.add_argument('--output_dir', type=str, default='infer_output')
parser.add_argument('--n_cond_frames', type=int, default=1) # 1 means use it , while 0 not 

# store true
parser.add_argument('--image', action='store_true')
args = parser.parse_args()
use_image = args.image

n = args.n
n_cond_frames = args.n_cond_frames
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

if not os.path.exists(args.ckpt):
    gpt = load_videogpt(args.ckpt)
else:
    gpt = VideoGPT.load_from_checkpoint(args.ckpt)
gpt = gpt.cuda()
gpt.eval()
args = gpt.hparams['args']
if use_image:
    real_images = output_dir / "real_images"
    generated_images = output_dir / "generated_images"
    real_images.mkdir(exist_ok=True)
    generated_images.mkdir(exist_ok=True)
args.batch_size = n
args.n_cond_frames = n_cond_frames
data = VideoData(args)
loader = data.test_dataloader()
for idx, batch in enumerate(tqdm(loader)):
    batch = {k: v.cuda() for k, v in batch.items()}
    real_videos = batch['video']
    real_videos = torch.clamp(real_videos, -0.5, 0.5) + 0.5
    samples = gpt.sample(n, batch)
    if use_image:
        # 为这个批次创建子目录
        real_batch_dir = real_images / f"batch_{idx}"
        generated_batch_dir = generated_images / f"batch_{idx}"
        real_batch_dir.mkdir(exist_ok=True, parents=True)
        generated_batch_dir.mkdir(exist_ok=True, parents=True)
        # 保存真实视频的帧
        for i in range(real_videos.size(0)):  # 遍历批次大小
            for t in range(real_videos.size(2)):  # 遍历时间维度

                frame = real_videos[i, :, t, :, :]
                frame_pil = Image.fromarray(
                    (frame.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                )
                frame_pil.save(real_batch_dir / f"video_{i}_frame_{t:03d}.png")
        # 保存生成的样本帧
        #        return samples # BCTHW

        for i in range(samples.size(0)):  # 遍历批次大小
            for t in range(samples.size(2)):  # 遍历时间维度
                frame = samples[i, :, t, :, :]
                frame_pil = Image.fromarray(
                    (frame.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                )
                frame_pil.save(generated_batch_dir / f"video_{i}_frame_{t:03d}.png")
    else:
        save_video_grid(real_videos, 'real_videos.gif')
        save_video_grid(samples, 'samples.gif')