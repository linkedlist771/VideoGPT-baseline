import argparse
import os
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
import imageio.v3 as iio
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from videogpt import VideoData, VideoSimVP, load_videogpt
from videogpt.data import label_maps, preprocess
from videogpt.utils import save_video_grid

def calculate_step_errors(real_frames, predicted_frames):
    """
    Calculate MAE and MSE for each prediction step.
    
    Args:
        real_frames: List of real frame tensors
        predicted_frames: List of predicted frame tensors
        
    Returns:
        Dictionary containing lists of MAE and MSE for each step
    """
    step_errors = {
        'mae': [],
        'mse': []
    }
    
    for real, pred in zip(real_frames, predicted_frames):
        # Convert tensors to numpy arrays if they aren't already
        if isinstance(real, torch.Tensor):
            real = real.cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
            
        # Ensure both are in the same range [0, 1] or [0, 255]
        if real.max() > 1.0:
            real = real / 255.0
        if pred.max() > 1.0:
            pred = pred / 255.0
            
        # Calculate errors
        mae = mean_absolute_error(real.flatten(), pred.flatten())
        mse = mean_squared_error(real.flatten(), pred.flatten())
        
        step_errors['mae'].append(mae)
        step_errors['mse'].append(mse)
    
    return step_errors

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str)
# input_video_path
#   --n_cond_frames 8 \
#   --n_pred_frames 8 \
parser.add_argument("--n_cond_frames", type=int, default=8)
parser.add_argument("--n_pred_frames", type=int, default=8)

parser.add_argument("--input_video_path", type=str) # 输入的是哪个mp4文件
parser.add_argument("--n", type=int, default=8)
parser.add_argument("--output_dir", type=str, default="infer_output")

args = parser.parse_args()
n = args.n
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
model = VideoSimVP.load_from_checkpoint(args.ckpt)
model = model.cuda()
model.eval()

input_video_path = Path(args.input_video_path)

# Read frames from the MP4 file
images_list = [frame for frame in iio.imiter(input_video_path)] # List of (H, W, C) uint8 numpy arrays

# Preprocess the images
# Convert numpy arrays to tensors and stack them into a single video tensor (T,H,W,C)
video_tensor = torch.stack([torch.from_numpy(frame) for frame in images_list], dim=0)
# Apply preprocessing with specified resolution (returns CTHW format)
resolution = 128  # Set your desired resolution here
processed_video = preprocess(video_tensor, resolution)
# Convert back to list of processed frames if needed
processed_frames = [processed_video[:,i].permute(1,2,0) for i in range(processed_video.shape[1])]

## Prepare the input data for the sampling....
video_filename_stem = input_video_path.stem
# Assuming the mp4 filename stem is a valid key or you have a default
label = label_maps.get(video_filename_stem) # Use .get for safety

image_length = len(processed_frames)

# Use hyperparameters from the loaded SimVP model
cond_len = args.n_cond_frames
pred_len = args.n_pred_frames

if image_length < cond_len + pred_len:
    raise ValueError(f"Video is too short. Need at least {cond_len + pred_len} frames, but got {image_length}.")

input_images_np = np.stack(processed_frames[0:cond_len]) # Shape: (cond_len, H, W, C)

batch_input_data = {
    "video": torch.from_numpy(input_images_np).to("cuda"), # Model's sample method should handle np array input
    "label": label,
}
# output_predicted_images_tensor shape: (n, pred_len, C, H, W), range [-0.5, 0.5]
output_predicted_images_tensor = model.long_seq_sample(n, batch_input_data)

# --- Saving logic ---
real_output_dir = output_dir / "real_frames"
predicted_output_dir = output_dir / "predicted_frames"
real_output_dir.mkdir(parents=True, exist_ok=True)
predicted_output_dir.mkdir(parents=True, exist_ok=True)

# Save real frames (ground truth for the prediction)
real_frames_to_save = processed_frames[cond_len: n+cond_len]
real_frames_np = []
for t, frame_tensor in enumerate(real_frames_to_save):
    # Convert tensor to numpy array and ensure values are in 0-255 range
    frame_np = ((frame_tensor.cpu().numpy() + 0.5) * 255).astype(np.uint8)
    real_frames_np.append(frame_np)
    img = Image.fromarray(frame_np)
    img.save(real_output_dir / f"real_frame_{t:03d}.png")
    if t == 0: # Log shape of first saved real frame for verification
        print(f"Saved real_frame_000.png with shape: {frame_np.shape}")

# The model returns a list of tensors - handle accordingly
print(f"Model returned {len(output_predicted_images_tensor)} tensors")

# Store predicted frames for error calculation
predicted_frames_np = []

# Create a directory for each sample in the list
for sample_idx, sample_tensor in enumerate(output_predicted_images_tensor):
    # Print shape information for debugging
    print(f"Processing sample tensor with shape: {sample_tensor.shape}")
    
    # Convert from tensor to numpy
    sample_np = sample_tensor.cpu().numpy()
    
    # The tensor shape is [B, C, T, H, W]
    # We need to iterate over T (frames) and convert to [H, W, C] for PIL
    num_frames = sample_np.shape[2]
    print(f"Number of frames to process: {num_frames}")
    
    for t in range(num_frames):  # Iterate over frames (T dimension)
        # Extract frame t for all batches and channels: [B, C, H, W]
        frame = sample_np[:, :, t, :, :]
        
        # Get first batch (if batched): [C, H, W]
        if frame.shape[0] > 1:
            frame = frame[0]
        else:
            frame = frame[0]  # Still need to remove the batch dim if it's 1
            
        # Convert from [C, H, W] to [H, W, C] for PIL
        frame_hwc = np.transpose(frame, (1, 2, 0))
        
        # Ensure we have uint8 data in the range 0-255
        frame_hwc = (frame_hwc * 255.0).astype(np.uint8)
        predicted_frames_np.append(frame_hwc)
        
        # Debug print first frame shape
        if sample_idx == 0 and t == 0:
            print(f"First frame shape after processing: {frame_hwc.shape}, dtype: {frame_hwc.dtype}")
        
        # Calculate the global frame index based on sample_idx and current frame t
        global_frame_idx = sample_idx * num_frames + t
        
        # Save the image directly in predicted_output_dir
        img = Image.fromarray(frame_hwc)
        img.save(predicted_output_dir / f"predicted_frame_{global_frame_idx:03d}.png")
        if sample_idx == 0 and t == 0:
            print(f"Saved predicted_frame_000.png with shape: {frame_hwc.shape}")

# Calculate and print errors for each step
step_errors = calculate_step_errors(real_frames_np, predicted_frames_np)

print("\nError metrics for each prediction step:")
print("Step\tMAE\t\tMSE")
print("-" * 40)
for step, (mae, mse) in enumerate(zip(step_errors['mae'], step_errors['mse'])):
    print(f"{step:02d}\t{mae:.6f}\t{mse:.6f}")
print("\nAverage MAE:", np.mean(step_errors['mae']))
print("Average MSE:", np.mean(step_errors['mse']))

print(f"Saved real frames to: {real_output_dir}")
print(f"Saved predicted frames to: {predicted_output_dir}")
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
