import argparse
import os
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
import imageio.v3 as iio
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from videogpt import VideoData, VideoGPT, load_videogpt
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
parser.add_argument("--n_cond_frames", type=int, default=8)
parser.add_argument("--n_pred_frames", type=int, default=8)
parser.add_argument("--input_video_path", type=str)
parser.add_argument("--n", type=int, default=8)  # Total frames to generate
parser.add_argument("--output_dir", type=str, default="infer_output")

args = parser.parse_args()
n = args.n
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# Load VideoGPT model
if not os.path.exists(args.ckpt):
    model = load_videogpt(args.ckpt)
else:
    model = VideoGPT.load_from_checkpoint(args.ckpt)
model = model.cuda()
model.eval()

# Set the prediction parameters
model.args.n_pred_frames = args.n_pred_frames

input_video_path = Path(args.input_video_path)

# Read frames from the MP4 file
images_list = [frame for frame in iio.imiter(input_video_path)]

# Preprocess the images following VideoGPT data format
video_tensor = torch.stack([torch.from_numpy(frame) for frame in images_list], dim=0)
resolution = 128
# preprocess returns CTHW format
processed_video = preprocess(video_tensor, resolution)  

# Prepare the input data following the exact format from sample_videogpt.py
video_filename_stem = input_video_path.stem
label = label_maps.get(video_filename_stem)

cond_len = args.n_cond_frames
if processed_video.shape[1] < cond_len:
    raise ValueError(f"Video has only {processed_video.shape[1]} frames, need at least {cond_len} conditioning frames.")

# Extract conditioning frames and add batch dimension: CTHW -> BCTHW
conditioning_video = processed_video[:, :cond_len, :, :].unsqueeze(0)  # Add batch dimension

batch_input_data = {
    "video": conditioning_video.cuda(),
    "label": label,
}

print(f"Input video shape: {conditioning_video.shape}")
print(f"Generating {n} total frames using {cond_len} conditioning frames")

# Generate long sequence
output_predicted_tensors = model.long_seq_sample(n, batch_input_data)

# --- Saving logic ---
real_output_dir = output_dir / "real_frames"
predicted_output_dir = output_dir / "predicted_frames"
real_output_dir.mkdir(parents=True, exist_ok=True)
predicted_output_dir.mkdir(parents=True, exist_ok=True)

# Save real frames (ground truth frames that we can compare with)
# Use frames from the original video that come after the conditioning frames
if processed_video.shape[1] > cond_len:
    real_frames_to_save = min(n, processed_video.shape[1] - cond_len)
    real_frames_np = []
    
    for t in range(real_frames_to_save):
        # Extract frame from CTHW format
        frame = processed_video[:, cond_len + t, :, :]  # Shape: [C, H, W]
        frame_np = ((frame.permute(1, 2, 0).cpu().numpy() + 0.5) * 255).astype(np.uint8)
        real_frames_np.append(frame_np)
        img = Image.fromarray(frame_np)
        img.save(real_output_dir / f"real_frame_{t:03d}.png")
        if t == 0:
            print(f"Saved real_frame_000.png with shape: {frame_np.shape}")
else:
    print("Warning: No real frames available for comparison (video too short)")
    real_frames_np = []

# Process and save predicted frames
print(f"Model returned {len(output_predicted_tensors)} tensor segments")

predicted_frames_np = []
global_frame_idx = 0

for segment_idx, segment_tensor in enumerate(output_predicted_tensors):
    print(f"Processing segment {segment_idx} with shape: {segment_tensor.shape}")
    
    # Convert to numpy: segment_tensor is BCTHW
    segment_np = segment_tensor.cpu().numpy()
    
    # Extract dimensions
    batch_size, channels, num_frames, height, width = segment_np.shape
    print(f"Segment {segment_idx}: batch_size={batch_size}, channels={channels}, frames={num_frames}")
    
    # Process each frame in the segment
    for t in range(num_frames):
        # Extract frame for first batch: [C, H, W]
        frame = segment_np[0, :, t, :, :]  # Take first batch
        
        # Convert from [C, H, W] to [H, W, C] for PIL
        frame_hwc = np.transpose(frame, (1, 2, 0))
        
        # Ensure we have uint8 data in the range 0-255
        frame_hwc = (frame_hwc * 255.0).astype(np.uint8)
        predicted_frames_np.append(frame_hwc)
        
        # Debug print for first frame
        if segment_idx == 0 and t == 0:
            print(f"First frame shape after processing: {frame_hwc.shape}, dtype: {frame_hwc.dtype}")
        
        # Save the image
        img = Image.fromarray(frame_hwc)
        img.save(predicted_output_dir / f"predicted_frame_{global_frame_idx:03d}.png")
        
        if segment_idx == 0 and t == 0:
            print(f"Saved predicted_frame_000.png with shape: {frame_hwc.shape}")
            
        global_frame_idx += 1

# Calculate and print errors for each step
if len(predicted_frames_np) > 0 and len(real_frames_np) > 0:
    # Ensure we compare the same number of frames
    min_frames = min(len(real_frames_np), len(predicted_frames_np))
    step_errors = calculate_step_errors(real_frames_np[:min_frames], predicted_frames_np[:min_frames])
    
    print("\nError metrics for each prediction step:")
    print("Step\tMAE\t\tMSE")
    print("-" * 40)
    for step, (mae, mse) in enumerate(zip(step_errors['mae'], step_errors['mse'])):
        print(f"{step:02d}\t{mae:.6f}\t{mse:.6f}")
    print("\nAverage MAE:", np.mean(step_errors['mae']))
    print("Average MSE:", np.mean(step_errors['mse']))

print(f"Saved real frames to: {real_output_dir}")
print(f"Saved predicted frames to: {predicted_output_dir}")
print(f"Total predicted frames: {global_frame_idx}") 