# this file doesn't have to depend on other files nor modules.
import cv2
from pathlib import Path 
from loguru import logger 
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

""""
Let me think about it, how many plots it can made for the long sequence prediction.


1. error comparsion between the true image and the predicted image in the error with the error image.(with in the 
training length, mark the generation with training length)


2. long sequence generation for different type of the images as input, mark the generation within the dataset 
and beyond the dataset.




"""



def load_image_list(image_dir: Path) -> tuple:
    """Load images from directory and return sorted images and their paths.
    
    Args:
        image_dir (Path): Directory containing images
        
    Returns:
        tuple: (sorted image paths, sorted images)
    """
    image_dict = {}
    for image_path in image_dir.glob("*.png"):
        # name split with _ take the last part, int it and sort it.
        image_name = int(image_path.stem.split("_")[-1])
        image = cv2.imread(str(image_path))
        image_dict[image_name] = (image_path, image)
    
    # Sort by frame number
    sorted_items = sorted(image_dict.items())
    sorted_paths = [item[1][0] for item in sorted_items]
    sorted_images = [item[1][1] for item in sorted_items]
    logger.debug(f"sorted_images: {len(sorted_images)}")
    
    return sorted_paths, sorted_images








ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs"

long_seq_dir = OUTPUT_DIR / "long_seq_out"

deposition_dir = long_seq_dir / "deposition"

etching_dir = long_seq_dir / "etching"




logger.debug(f"ROOT: {ROOT}")

def visualize_prediction_error(true_images: list, pred_images: list, training_length: int, output_path: Path):
    """Visualize prediction error between true and predicted images.
    
    Args:
        true_images (list): List of ground truth images
        pred_images (list): List of predicted images
        training_length (int): Number of frames used in training
        output_path (Path): Path to save visualization
    """
    # Ensure we have valid images
    true_images = [img for img in true_images if img is not None and img.size > 0]
    pred_images = [img for img in pred_images if img is not None and img.size > 0]
    
    min_length = min(len(true_images), len(pred_images))
    true_images = true_images[:min_length]
    pred_images = pred_images[:min_length]
    
    num_frames = len(true_images)
    logger.debug(f"Initial num_frames: {num_frames}")
    
    # Ensure we have at least one frame
    if num_frames == 0:
        logger.error("No frames to visualize")
        return
    
    # Fix the number of frames to show at 9, or less if we have fewer frames
    target_frames = min(9, num_frames)
    
    # Sample frames evenly from the entire sequence
    if num_frames > target_frames:
        # Always include first and last frame
        if target_frames >= 2:
            indices = [0]  # Start with the first frame
            
            # Calculate step size to distribute remaining frames evenly
            step = (num_frames - 1) / (target_frames - 1)
            
            # Add middle frames
            for i in range(1, target_frames - 1):
                idx = int(i * step)
                indices.append(idx)
                
            indices.append(num_frames - 1)  # End with the last frame
        else:
            # If we only want 1 frame, take the middle one
            indices = [num_frames // 2]
            
        true_images = [true_images[i] for i in indices]
        pred_images = [pred_images[i] for i in indices]
        
        # For each index, keep track of the original frame number
        frame_indices = indices
        num_frames = len(indices)
    else:
        # If we have fewer frames than target, use all frames
        frame_indices = list(range(num_frames))
    
    logger.debug(f"After sampling - num_frames: {num_frames}, selected indices: {frame_indices}")
    
    # Calculate figure dimensions with minimum sizes
    base_height = 4  # height per row
    min_width_per_frame = 3
    
    # Ensure at least one frame width and reasonable height
    fig_width = max(min_width_per_frame * max(1, num_frames), 8)
    fig_height = max(base_height * 3, 12)  # 3 rows: true, predicted, error
    
    # Cap maximum dimensions
    fig_width = min(fig_width, 32)
    
    # Ensure dimensions are positive
    fig_width = max(1, fig_width)
    fig_height = max(1, fig_height)
    
    logger.debug(f"Figure dimensions - width: {fig_width}, height: {fig_height}")
    
    import matplotlib.pyplot as plt
    try:
        fig = plt.figure(figsize=(fig_width, fig_height))
        if fig is None:
            logger.error("Failed to create figure")
            return
        
        # Create a common y-label for each row
        fig.text(0.02, 0.7, 'Ground Truth', fontsize=24, fontweight='bold', rotation=90, va='center')
        fig.text(0.02, 0.5, 'Prediction', fontsize=24, fontweight='bold', rotation=90, va='center')
        fig.text(0.02, 0.25, 'Error', fontsize=24, fontweight='bold', rotation=90, va='center')
            
        for i in range(num_frames):
            # True image
            ax1 = plt.subplot(3, max(1, num_frames), i + 1)
            if true_images[i] is not None and true_images[i].size > 0:
                ax1.imshow(cv2.cvtColor(true_images[i], cv2.COLOR_BGR2RGB))
            ax1.axis('off')
            frame_num = frame_indices[i]  # Use the original frame number
            if frame_num < training_length:
                ax1.set_title(f'Frame {frame_num}\n(Training)', fontsize=24, color='blue')
                # Add a blue border to training frames
                for spine in ax1.spines.values():
                    spine.set_visible(True)
                    spine.set_color('blue')
                    spine.set_linewidth(1.5)
            else:
                ax1.set_title(f'Frame {frame_num}\n(Extrapolation)', fontsize=24, color='red')
                # Add a red border to generation frames
                for spine in ax1.spines.values():
                    spine.set_visible(True)
                    spine.set_color('red')
                    spine.set_linewidth(1.5)
                
            # Predicted image
            ax2 = plt.subplot(3, max(1, num_frames), num_frames + i + 1)
            if pred_images[i] is not None and pred_images[i].size > 0:
                ax2.imshow(cv2.cvtColor(pred_images[i], cv2.COLOR_BGR2RGB))
            ax2.axis('off')
            # Apply same coloring to prediction frames
            if frame_num < training_length:
                for spine in ax2.spines.values():
                    spine.set_visible(True)
                    spine.set_color('blue')
                    spine.set_linewidth(1.5)
            else:
                for spine in ax2.spines.values():
                    spine.set_visible(True)
                    spine.set_color('red')
                    spine.set_linewidth(1.5)
                
            # Error image
            ax3 = plt.subplot(3, max(1, num_frames), 2*num_frames + i + 1)
            if true_images[i] is not None and pred_images[i] is not None and true_images[i].size > 0 and pred_images[i].size > 0:
                error = cv2.absdiff(true_images[i], pred_images[i])
                ax3.imshow(cv2.cvtColor(error, cv2.COLOR_BGR2RGB))
            ax3.axis('off')
            # Apply same coloring to error frames
            if frame_num < training_length:
                for spine in ax3.spines.values():
                    spine.set_visible(True)
                    spine.set_color('blue')
                    spine.set_linewidth(1.5)
            else:
                for spine in ax3.spines.values():
                    spine.set_visible(True)
                    spine.set_color('red')
                    spine.set_linewidth(1.5)
            
        plt.tight_layout(rect=[0.03, 0, 1, 1])  # Add left margin for row labels
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        logger.info(f"Saved error visualization to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        logger.debug(f"Error details - num_frames: {num_frames}, fig_width: {fig_width}, fig_height: {fig_height}")
        plt.close('all')

def visualize_long_sequence(images: list, training_length: int, output_path: Path, dataset_size: int = None, num_cols: int = 8):
    """Visualize long sequence generation with training/generation boundary marked.
    
    Args:
        images (list): List of images in sequence
        training_length (int): Number of frames used in training
        output_path (Path): Path to save visualization
        dataset_size (int, optional): Total number of frames in the original dataset. 
                                     Used to mark separation between in-dataset and beyond-dataset frames.
                                     If None, only training length is marked. Defaults to None.
        num_cols (int, optional): Number of columns in grid. Defaults to 8.
    """
    # Ensure we have valid images
    images = [img for img in images if img is not None and img.size > 0]
    
    num_frames = len(images)
    logger.debug(f"Initial num_frames: {num_frames}")
    
    # Ensure we have at least one frame
    if num_frames == 0:
        logger.error("No frames to visualize")
        return
    
    # If dataset_size is not provided, use training_length as default
    if dataset_size is None:
        dataset_size = training_length
    
    # We want to select images that represent both training and generation periods
    # Split images into training, in-dataset generation, and beyond-dataset generation
    training_frames = []
    in_dataset_frames = []  # Frames after training but still within dataset
    beyond_dataset_frames = []  # Frames beyond dataset
    
    for i, img in enumerate(images):
        if i < training_length:
            training_frames.append((i, img))
        elif i < dataset_size:
            in_dataset_frames.append((i, img))
        else:
            beyond_dataset_frames.append((i, img))
    
    # Sample training frames
    sampled_training = []
    if training_frames:
        target_training_samples = min(3, len(training_frames))
        step = max(1, len(training_frames) // target_training_samples)
        indices = list(range(0, len(training_frames), step))
        if indices and indices[-1] != len(training_frames) - 1:
            indices.append(len(training_frames) - 1)
        sampled_training = [training_frames[i] for i in indices[:target_training_samples]]
    
    # Sample in-dataset generation frames
    sampled_in_dataset = []
    if in_dataset_frames:
        target_in_dataset_samples = min(3, len(in_dataset_frames))
        step = max(1, len(in_dataset_frames) // target_in_dataset_samples)
        indices = list(range(0, len(in_dataset_frames), step))
        if indices and indices[-1] != len(in_dataset_frames) - 1:
            indices.append(len(in_dataset_frames) - 1)
        sampled_in_dataset = [in_dataset_frames[i] for i in indices[:target_in_dataset_samples]]
    
    # Sample beyond-dataset generation frames
    sampled_beyond_dataset = []
    if beyond_dataset_frames:
        target_beyond_dataset_samples = min(3, len(beyond_dataset_frames))
        step = max(1, len(beyond_dataset_frames) // target_beyond_dataset_samples)
        indices = list(range(0, len(beyond_dataset_frames), step))
        if indices and indices[-1] != len(beyond_dataset_frames) - 1:
            indices.append(len(beyond_dataset_frames) - 1)
        sampled_beyond_dataset = [beyond_dataset_frames[i] for i in indices[:target_beyond_dataset_samples]]
    
    logger.debug(f"Selected {len(sampled_training)} training frames, {len(sampled_in_dataset)} in-dataset frames, "
                f"and {len(sampled_beyond_dataset)} beyond-dataset frames")
    
    # Import matplotlib and create figure
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Create a wide figure for horizontal layout
    fig = plt.figure(figsize=(18, 5))
    
    # Calculate total number of subplots and widths for each section
    total_plots = len(sampled_training) + len(sampled_in_dataset) + len(sampled_beyond_dataset)
    
    if total_plots == 0:
        logger.error("No frames to visualize after sampling")
        return
    
    # Calculate relative widths for each section
    training_width = len(sampled_training) / total_plots if total_plots > 0 else 0
    in_dataset_width = len(sampled_in_dataset) / total_plots if total_plots > 0 else 0
    beyond_dataset_width = len(sampled_beyond_dataset) / total_plots if total_plots > 0 else 0
    
    # Create GridSpec with three sections horizontally if we have all three types
    if sampled_training and sampled_in_dataset and sampled_beyond_dataset:
        gs = GridSpec(1, 3, width_ratios=[len(sampled_training), len(sampled_in_dataset), len(sampled_beyond_dataset)], wspace=0.05)
    elif sampled_training and (sampled_in_dataset or sampled_beyond_dataset):
        # Two sections only
        second_section = sampled_in_dataset if sampled_in_dataset else sampled_beyond_dataset
        gs = GridSpec(1, 2, width_ratios=[len(sampled_training), len(second_section)], wspace=0.05)
    else:
        # Single section
        gs = GridSpec(1, 1)
    
    # Positions for section titles and separators
    section_positions = []
    
    # Add section title for training frames
    if sampled_training:
        section_positions.append((training_width/2, 'Generation within the training length', 'blue'))
    
    # Add section title for in-dataset frames after training
    if sampled_in_dataset:
        in_dataset_center = training_width + in_dataset_width/2
        section_positions.append((in_dataset_center, 'Generation in dataset length', 'green'))
    
    # Add section title for beyond-dataset frames
    if sampled_beyond_dataset:
        beyond_dataset_center = training_width + in_dataset_width + beyond_dataset_width/2
        section_positions.append((beyond_dataset_center, 'Generation beyond the dataset length', 'red'))
    
    # Add all section titles
    for pos, title, color in section_positions:
        fig.text(pos, 0.95, title, ha='center', fontsize=12, color=color, fontweight='bold')
    
    # Current position counter for subplot placement
    current_pos = 0
    
    # Training section (left)
    for i, (frame_idx, img) in enumerate(sampled_training):
        current_pos += 1
        ax = plt.subplot(1, total_plots, current_pos)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f'Frame {frame_idx}', fontsize=9)
        # Add blue border to training frames
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('blue')
            spine.set_linewidth(1.5)
    
    # Add a vertical separator between training and in-dataset section
    if sampled_training and (sampled_in_dataset or sampled_beyond_dataset):
        ax_separator = fig.add_axes([0, 0, 1, 1])
        ax_separator.axvline(x=training_width, color='black', linestyle='--', linewidth=2)
        ax_separator.axis('off')
    
    # In-dataset section (middle)
    for i, (frame_idx, img) in enumerate(sampled_in_dataset):
        current_pos += 1
        ax = plt.subplot(1, total_plots, current_pos)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f'Frame {frame_idx}', fontsize=9)
        # Add blue border to in-dataset frames
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('blue')
            spine.set_linewidth(1.5)
    
    # Add a vertical separator between in-dataset and beyond-dataset section
    if sampled_in_dataset and sampled_beyond_dataset:
        beyond_dataset_start = training_width + in_dataset_width
        ax_dataset_separator = fig.add_axes([0, 0, 1, 1])
        ax_dataset_separator.axvline(x=beyond_dataset_start, color='green', linestyle='--', linewidth=2)
        ax_dataset_separator.axis('off')
        # Add a label for the separator
        # fig.text(beyond_dataset_start, 0.85, 'Dataset\nBoundary', ha='center', va='center', 
        #          fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Beyond-dataset section (right)
    for i, (frame_idx, img) in enumerate(sampled_beyond_dataset):
        current_pos += 1
        ax = plt.subplot(1, total_plots, current_pos)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f'Frame {frame_idx}', fontsize=9)
        # Add red border to beyond-dataset frames
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('red')
            spine.set_linewidth(1.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Add space for the titles
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved long sequence visualization to {output_path}")

def main():
    # Load true and predicted images for both deposition and etching
    true_deposition_paths, true_deposition_images = load_image_list(deposition_dir / "real_frames")
    pred_deposition_paths, pred_deposition_images = load_image_list(deposition_dir / "predicted_frames")
    
    true_etching_paths, true_etching_images = load_image_list(etching_dir / "real_frames")
    pred_etching_paths, pred_etching_images = load_image_list(etching_dir / "predicted_frames")
    
    # Create output directories if they don't exist
    (long_seq_dir / "visualizations").mkdir(exist_ok=True, parents=True)
    
    # Visualize prediction errors
    training_length = 16  # Adjust this based on your actual training length
    deposition_dataset_size = 150
    etching_dataset_size = 50
    visualize_prediction_error(
        true_deposition_images, 
        pred_deposition_images,
        training_length,
        long_seq_dir / "visualizations" / "deposition_error.png",
    )
    
    visualize_prediction_error(
        true_etching_images,
        pred_etching_images,
        training_length,
        long_seq_dir / "visualizations" / "etching_error.png",
    )
    
    # Visualize long sequences
    visualize_long_sequence(
        pred_deposition_images,
        training_length,
        long_seq_dir / "visualizations" / "deposition_sequence.png",
        deposition_dataset_size
    )
    
    visualize_long_sequence(
        pred_etching_images,
        training_length,
        long_seq_dir / "visualizations" / "etching_sequence.png",
        etching_dataset_size
    )

if __name__ == "__main__":
    main()