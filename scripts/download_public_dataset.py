"""
Download publicly available deepfake dataset from Kaggle
No approval required - instant download
"""
import os
import sys
import subprocess
from pathlib import Path


def check_kaggle():
    """Check if Kaggle CLI is installed"""
    try:
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False


def install_kaggle():
    """Install Kaggle CLI"""
    print("\nInstalling Kaggle CLI...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'])
    print("✓ Kaggle CLI installed")


def setup_kaggle_credentials():
    """Guide user through Kaggle credentials setup"""
    print("\n" + "=" * 70)
    print("Kaggle API Setup Required")
    print("=" * 70)
    print("\nFollow these steps:")
    print("\n1. Go to: https://www.kaggle.com/")
    print("2. Sign in or create an account (free)")
    print("3. Click on your profile picture (top right)")
    print("4. Go to 'Account' → 'Settings'")
    print("5. Scroll down to 'API' section")
    print("6. Click 'Create New API Token'")
    print("7. A file 'kaggle.json' will download")
    print("8. Move it to: C:\\Users\\user\\.kaggle\\kaggle.json")
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    print(f"\nExpected location: {kaggle_json}")
    
    input("\nPress Enter after you've placed kaggle.json in the correct location...")
    
    if kaggle_json.exists():
        print("✓ kaggle.json found!")
        # Set permissions (required on Unix systems)
        try:
            os.chmod(kaggle_json, 0o600)
        except:
            pass
        return True
    else:
        print(f"✗ kaggle.json not found at {kaggle_json}")
        print("Please place the file there and run this script again")
        return False


def download_dataset(dataset_name, output_dir):
    """Download dataset from Kaggle"""
    print(f"\nDownloading {dataset_name}...")
    print("This may take 10-30 minutes depending on your internet speed")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', dataset_name,
            '-p', str(output_path),
            '--unzip'
        ], check=True)
        
        print(f"\n✓ Download complete!")
        print(f"  Location: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed: {e}")
        return False


def main():
    print("=" * 70)
    print("Public Deepfake Dataset Downloader")
    print("No approval required - instant download!")
    print("=" * 70)
    
    # Check Kaggle CLI
    if not check_kaggle():
        print("\n⚠ Kaggle CLI not found")
        install = input("Install Kaggle CLI now? (y/n): ").lower()
        if install == 'y':
            install_kaggle()
        else:
            print("Please install manually: pip install kaggle")
            sys.exit(1)
    
    # Setup credentials
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        if not setup_kaggle_credentials():
            sys.exit(1)
    
    # Choose dataset
    print("\n" + "=" * 70)
    print("Available Datasets")
    print("=" * 70)
    print("\n1. 140K Real and Fake Faces [RECOMMENDED]")
    print("   Size: ~2.5GB")
    print("   Images: 140,000")
    print("   Quality: High")
    print("   Time: ~10-20 minutes")
    print("   Dataset: xhlulu/140k-real-and-fake-faces")
    
    print("\n2. Deepfake Detection Sample")
    print("   Size: ~10GB")
    print("   Videos: 400+")
    print("   Quality: High")
    print("   Time: ~30-60 minutes")
    print("   Dataset: mrmorj/deepfake-detection-sample")
    
    print("\n3. DeepFake Detection Dataset (Images)")
    print("   Size: ~1GB")
    print("   Images: 10,000+")
    print("   Quality: Medium")
    print("   Time: ~5-10 minutes")
    print("   Dataset: manjilkarki/deepfake-detection-dataset")
    
    choice = input("\nChoose dataset (1/2/3): ").strip()
    
    datasets = {
        '1': ('xhlulu/140k-real-and-fake-faces', 'data/140k_faces'),
        '2': ('mrmorj/deepfake-detection-sample', 'data/dfdc_sample'),
        '3': ('manjilkarki/deepfake-detection-dataset', 'data/deepfake_images')
    }
    
    if choice not in datasets:
        print("Invalid choice")
        sys.exit(1)
    
    dataset_name, output_dir = datasets[choice]
    
    print(f"\nDownloading: {dataset_name}")
    print(f"Output: {output_dir}")
    
    proceed = input("\nProceed with download? (y/n): ").lower()
    if proceed != 'y':
        print("Download cancelled")
        sys.exit(0)
    
    # Download
    success = download_dataset(dataset_name, output_dir)
    
    if success:
        print("\n" + "=" * 70)
        print("Download Complete!")
        print("=" * 70)
        
        output_path = Path(output_dir)
        
        # Count files
        image_count = len(list(output_path.rglob('*.jpg'))) + \
                     len(list(output_path.rglob('*.png')))
        video_count = len(list(output_path.rglob('*.mp4'))) + \
                     len(list(output_path.rglob('*.avi')))
        
        print(f"\nDataset Statistics:")
        print(f"  Images: {image_count:,}")
        print(f"  Videos: {video_count:,}")
        print(f"  Location: {output_path.absolute()}")
        
        print("\n" + "=" * 70)
        print("Next Steps")
        print("=" * 70)
        
        if video_count > 0:
            print("\n1. Extract frames from videos:")
            print(f"   python scripts/extract_frames_from_videos.py \\")
            print(f"       --input {output_dir} \\")
            print(f"       --output {output_dir}_frames")
            print("\n2. Then prepare dataset:")
            print(f"   python scripts/prepare_dataset.py \\")
            print(f"       --input {output_dir}_frames \\")
            print("       --output data/processed")
        else:
            print("\n1. Prepare dataset:")
            print(f"   python scripts/prepare_dataset.py \\")
            print(f"       --input {output_dir} \\")
            print("       --output data/processed")
        
        print("\n2. Start training:")
        print("   python backend/training/train_efficientnet.py \\")
        print("       --data-dir data/processed \\")
        print("       --batch-size 32 \\")
        print("       --epochs 25")


if __name__ == '__main__':
    main()
