import argparse
from pathlib import Path
import shutil
import random
from tqdm import tqdm


def copy_mp4_files(src_dir, dst_dir):
    for file in src_dir.glob('*.mp4'):
        shutil.copy2(file, dst_dir)


def split_data(source_dir: Path, target_dir: Path, train_test_ratio: float = 0.8):
    # Create train and test directories in the target directory
    train_dir = target_dir / 'train'
    test_dir = target_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Get all immediate subdirectories in the source directory
    subdirs = [d for d in source_dir.iterdir() if d.is_dir()]

    # Shuffle the subdirectories randomly
    random.shuffle(subdirs)

    # Calculate the split index
    split_index = int(len(subdirs) * train_test_ratio)

    # Split subdirectories into train and test sets
    train_dirs = subdirs[:split_index]
    test_dirs = subdirs[split_index:]

    # Copy MP4 files from train directories
    for src_dir in tqdm(train_dirs, desc="Copying train MP4 files"):
        copy_mp4_files(src_dir, train_dir)

    # Copy MP4 files from test directories
    for src_dir in tqdm(test_dirs, desc="Copying test MP4 files"):
        copy_mp4_files(src_dir, test_dir)

    print(f"Data split completed. Train set: {train_dir}, Test set: {test_dir}")
    print(f"Number of train MP4 files: {len(list(train_dir.glob('*.mp4')))}")
    print(f"Number of test MP4 files: {len(list(test_dir.glob('*.mp4')))}")


def main():
    parser = argparse.ArgumentParser(description="Split data into train and test sets.")
    parser.add_argument("--source", type=Path, required=True, help="Source directory containing the data")
    parser.add_argument("--target", type=Path, required=True, help="Target directory for the split data")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train-test split ratio (default: 0.8)")
    args = parser.parse_args()

    split_data(args.source, args.target, args.ratio)


if __name__ == "__main__":
    main()
