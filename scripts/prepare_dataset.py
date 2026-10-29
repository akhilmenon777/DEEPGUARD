"""
Prepare deepfake dataset for training
Organizes images into train/val/test splits with real/fake classes
"""
import os
import sys
import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import cv2


def create_directory_structure(output_dir):
    """Create train/val/test directory structure"""
    splits = ['train', 'val', 'test']
    classes = ['real', 'fake']
    
    for split in splits:
        for cls in classes:
            path = Path(output_dir) / split / cls
            path.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directory structure at {output_dir}")


def collect_images(input_dir, real_subdir='real', fake_subdir='fake'):
    """Collect all image paths from input directory"""
    input_path = Path(input_dir)
    
    real_images = []
    fake_images = []
    
    # Find real images
    real_dir = input_path / real_subdir
    if real_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            real_images.extend(list(real_dir.rglob(ext)))
    
    # Find fake images
    fake_dir = input_path / fake_subdir
    if fake_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            fake_images.extend(list(fake_dir.rglob(ext)))
    
    print(f"\nFound {len(real_images)} real images")
    print(f"Found {len(fake_images)} fake images")
    
    return real_images, fake_images


def resize_and_copy(src_path, dst_path, target_size=224):
    """Resize image and copy to destination"""
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            return False
        
        # Resize
        img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # Save
        cv2.imwrite(str(dst_path), img_resized)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def split_and_organize(real_images, fake_images, output_dir, 
                      train_ratio=0.7, val_ratio=0.15, resize=True, target_size=224):
    """Split images into train/val/test and organize by class"""
    
    # Shuffle
    random.shuffle(real_images)
    random.shuffle(fake_images)
    
    # Calculate split indices
    def get_splits(images):
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
    
    real_splits = get_splits(real_images)
    fake_splits = get_splits(fake_images)
    
    output_path = Path(output_dir)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Process real images
        real_dst_dir = output_path / split / 'real'
        for i, src in enumerate(tqdm(real_splits[split], desc=f'{split}/real')):
            dst = real_dst_dir / f"real_{i:06d}{src.suffix}"
            if resize:
                resize_and_copy(src, dst, target_size)
            else:
                shutil.copy2(src, dst)
        
        # Process fake images
        fake_dst_dir = output_path / split / 'fake'
        for i, src in enumerate(tqdm(fake_splits[split], desc=f'{split}/fake')):
            dst = fake_dst_dir / f"fake_{i:06d}{src.suffix}"
            if resize:
                resize_and_copy(src, dst, target_size)
            else:
                shutil.copy2(src, dst)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Dataset Preparation Complete!")
    print("=" * 70)
    print("\nStatistics:")
    for split in ['train', 'val', 'test']:
        real_count = len(list((output_path / split / 'real').glob('*')))
        fake_count = len(list((output_path / split / 'fake').glob('*')))
        total = real_count + fake_count
        print(f"{split.capitalize()}:")
        print(f"  Real: {real_count:,} | Fake: {fake_count:,} | Total: {total:,}")


def main():
    parser = argparse.ArgumentParser(description='Prepare deepfake dataset for training')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing raw data')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for organized dataset')
    parser.add_argument('--real-dir', type=str, default='real',
                        help='Subdirectory name for real images')
    parser.add_argument('--fake-dir', type=str, default='fake',
                        help='Subdirectory name for fake images')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of training data')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Ratio of validation data')
    parser.add_argument('--resize', action='store_true', default=True,
                        help='Resize images to target size')
    parser.add_argument('--target-size', type=int, default=224,
                        help='Target image size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("=" * 70)
    print("Deepfake Dataset Preparation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Split: {args.train_ratio:.0%} train / {args.val_ratio:.0%} val / {1-args.train_ratio-args.val_ratio:.0%} test")
    print(f"  Resize: {args.resize}")
    if args.resize:
        print(f"  Target size: {args.target_size}x{args.target_size}")
    
    # Create directory structure
    create_directory_structure(args.output)
    
    # Collect images
    real_images, fake_images = collect_images(
        args.input, 
        real_subdir=args.real_dir,
        fake_subdir=args.fake_dir
    )
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("\n❌ Error: No images found!")
        print(f"   Check that {args.input}/{args.real_dir} and {args.input}/{args.fake_dir} exist")
        print(f"   and contain images (.jpg, .jpeg, .png)")
        sys.exit(1)
    
    # Split and organize
    split_and_organize(
        real_images, 
        fake_images, 
        args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        resize=args.resize,
        target_size=args.target_size
    )
    
    print(f"\n✓ Dataset ready for training at: {args.output}")
    print(f"\nNext step: Run training script")
    print(f"  python backend/training/train_efficientnet.py --data-dir {args.output}")


if __name__ == '__main__':
    main()
