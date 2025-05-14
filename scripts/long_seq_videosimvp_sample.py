import argparse
import os
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm

from videogpt import VideoData, VideoSimVP, load_videogpt
from videogpt.data import label_maps
from videogpt.utils import save_video_grid
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="ucf101_uncond_gpt")
parser.add_argument("--input_dir", type=str) # 输入的是哪个文件夹的数据，然后读取前面的控制帧， 然后作为输入。
parser.add_argument("--n", type=int, default=8)
parser.add_argument("--output_dir", type=str, default="infer_output")

args = parser.parse_args()
n = args.n
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)
model = VideoSimVP.load_from_checkpoint(args.ckpt)
model = model.cuda()
model.eval()

input_dir = Path(args.input_dir)

images_list = [i for i in input_dir.glob("*.png") if i.is_file()]
images_list = sorted(images_list, key=lambda x: int(x.stem.split("_")[-1]))
images_list = [np.array(Image.open(i)) for i in images_list]

## Prepare the input data for the sampling....
# PurePath.stem => directory name
directory_name = input_dir.stem
label = label_maps[directory_name]
# get all the images.
image_length = len(images_list)

input_images = np.vstack(images_list[0 : 0 + model.args.n_cond_frames])

# for i in range(model.args.n_cond_frames, image_length, model.args.n_cond_frames):
# batch_real_images = np.vstack(images_list[i : i + model.args.n_cond_frames])
output_real_images = [np.vstack(images_list[i : i + model.args.n_cond_frames]) for i in range(model.args.n_cond_frames, image_length, model.args.n_cond_frames)]
batch_input_data = {
    "video": input_images,
    "label": label,
}
output_predicted_images = model.sample(n, batch_input_data)
# 差不多这里完成了，后面再看看。继续怎么写后面的。






# iterate over all the images in this sub directory
# label_maps


# args = model.hparams["args"]


# images


# def long_seq_sample(self, n: int, batch=None):
#     # We need a batch of conditioning frames
#     assert batch is not None, "Batch must be provided for conditioning"
#     x = batch["video"]
#     labels = batch["label"]
#


# args.batch_size = n
# data = VideoData(args)
# loader = data.test_dataloader()
# for idx, batch in enumerate(tqdm(loader)):
#     # from loguru import logger
#
#     # logger.info(f"batch:\n {batch}")
#     # batch = {k: v.cuda() for k, v in batch.items()}
#     batch = {
#         "video": batch["video"].cuda(),
#         "label": batch["label"],  # label this is list[str]
#     }
#     real_videos = batch["video"]
#     real_videos = torch.clamp(real_videos, -0.5, 0.5) + 0.5
#     # size...
#     samples = gpt.sample(n, batch)  # for simvp it is a littel bit different
#     # the batch has both the input and the target, we should only use
#     # the target and the predicted to eval.
#     if use_image:
#         # 为这个批次创建子目录
#         real_batch_dir = real_images / f"batch_{idx}"
#         generated_batch_dir = generated_images / f"batch_{idx}"
#         real_batch_dir.mkdir(exist_ok=True, parents=True)
#         generated_batch_dir.mkdir(exist_ok=True, parents=True)
#         # 保存真实视频的帧
#         for i in range(real_videos.size(0)):  # 遍历批次大小
#             for t in range(real_videos.size(2)):  # 遍历时间维度
#                 skip = real_videos.size(2) // 2
#                 if t >= skip:
#                     frame = real_videos[i, :, t, :, :]
#                     frame_pil = Image.fromarray(
#                         (frame.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
#                     )
#                     frame_pil.save(
#                         real_batch_dir / f"video_{i}_frame_{(t-skip):03d}.png"
#                     )
#         # 保存生成的样本帧
#         #        return samples # BCTHW
#         # we just sampled the later part of the predictedc
#         for i in range(samples.size(0)):  # 遍历批次大小
#             for t in range(samples.size(2)):  # 遍历时间维度
#                 frame = samples[i, :, t, :, :]
#                 frame_pil = Image.fromarray(
#                     (frame.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
#                 )
#                 frame_pil.save(generated_batch_dir / f"video_{i}_frame_{t:03d}.png")
#     else:
#         save_video_grid(real_videos, "real_videos.gif")
#         save_video_grid(samples, "samples.gif")
