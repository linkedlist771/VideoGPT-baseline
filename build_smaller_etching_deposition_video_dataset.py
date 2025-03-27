from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import random
import shutil

def build_smaller_video_dataset(source_dir: Path, target_dir: Path, ratio: float=0.1):
    # Set fixed random seed for reproducibility
    random.seed(42)
    
    source_train_dir = source_dir / "train"
    source_test_dir = source_dir / "test"
    target_train_dir = target_dir / "train"
    target_test_dir = target_dir / "test"
    target_train_dir.mkdir(exist_ok=True, parents=True)
    target_test_dir.mkdir(exist_ok=True, parents=True)    
    train_mp4_files = list(source_train_dir.glob("*.mp4"))
    test_mp4_files = list(source_test_dir.glob("*.mp4"))
    
    # Randomly select files based on ratio
    num_train_files = int(len(train_mp4_files) * ratio)
    num_test_files = int(len(test_mp4_files) * ratio)
    
    selected_train_files = random.sample(train_mp4_files, num_train_files)
    selected_test_files = random.sample(test_mp4_files, num_test_files)
    
    # Copy selected files to target directories
    for file in tqdm(selected_train_files, desc="Copying train files"):
        shutil.copy2(file, target_train_dir / file.name)
    
    for file in tqdm(selected_test_files, desc="Copying test files"):
        shutil.copy2(file, target_test_dir / file.name)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory containing train and test folders")
    parser.add_argument("--target_dir", type=str, required=True, help="Target directory to save the smaller dataset")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio of files to copy (default: 0.1)")
    
    args = parser.parse_args()
    build_smaller_video_dataset(Path(args.source_dir), Path(args.target_dir), args.ratio) 