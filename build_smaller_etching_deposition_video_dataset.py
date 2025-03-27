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
    
    # Get files for each type
    train_mp4_files = list(source_train_dir.glob("*.mp4"))
    test_mp4_files = list(source_test_dir.glob("*.mp4"))

    sub_prefixes_mp4_files_train = [
            file for file in train_mp4_files if file.stem.startswith("sub")]
    sub_prefixes_mp4_files_test = [
            file for file in test_mp4_files if file.stem.startswith("sub")]
    
    process_prefixes_mp4_files_train = [
            file for file in train_mp4_files if file.stem.startswith("process")]
    process_prefixes_mp4_files_test = [
            file for file in test_mp4_files if file.stem.startswith("process")]
    
    # Calculate number of files to select for each type
    num_train_sub = int(len(sub_prefixes_mp4_files_train) * ratio)
    num_train_process = int(len(process_prefixes_mp4_files_train) * ratio)
    num_test_sub = int(len(sub_prefixes_mp4_files_test) * ratio)
    num_test_process = int(len(process_prefixes_mp4_files_test) * ratio)
    
    # Randomly select files for each type
    selected_train_sub = random.sample(sub_prefixes_mp4_files_train, num_train_sub)
    selected_train_process = random.sample(process_prefixes_mp4_files_train, num_train_process)
    selected_test_sub = random.sample(sub_prefixes_mp4_files_test, num_test_sub)
    selected_test_process = random.sample(process_prefixes_mp4_files_test, num_test_process)
    
    # Copy selected files to target directories
    for file in tqdm(selected_train_sub, desc="Copying train sub files"):
        shutil.copy2(file, target_train_dir / file.name)
    
    for file in tqdm(selected_train_process, desc="Copying train process files"):
        shutil.copy2(file, target_train_dir / file.name)
    
    for file in tqdm(selected_test_sub, desc="Copying test sub files"):
        shutil.copy2(file, target_test_dir / file.name)
    
    for file in tqdm(selected_test_process, desc="Copying test process files"):
        shutil.copy2(file, target_test_dir / file.name)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory containing train and test folders")
    parser.add_argument("--target_dir", type=str, required=True, help="Target directory to save the smaller dataset")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio of files to copy (default: 0.1)")
    
    args = parser.parse_args()
    build_smaller_video_dataset(Path(args.source_dir), Path(args.target_dir), args.ratio) 