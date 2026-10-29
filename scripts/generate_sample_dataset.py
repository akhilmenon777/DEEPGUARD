"""
Generate a small sample dataset for testing training pipeline
Creates synthetic "real" and "fake" images with distinguishing features
"""
import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def generate_real_image(size=224):
    """
    Generate synthetic "real" image with natural characteristics:
    - Smooth gradients
    - Natural colors
    - Normal texture
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create smooth gradient background
    for i in range(size):
        for j in range(size):
            r = int(120 + 30 * np.sin(i / size * np.pi))
            g = int(150 + 20 * np.cos(j / size * np.pi))
            b = int(180 - 30 * np.sin((i+j) / size * np.pi))
            img[i, j] = [b, g, r]
    
    # Add natural-looking face shape
    center = (size // 2, size // 2)
    axes = (size // 3, size // 2)
    cv2.ellipse(img, center, axes, 0, 0, 360, (220, 200, 180), -1)
    
    # Add features (eyes, nose, mouth)
    # Eyes
    cv2.circle(img, (size//3, size//3), 8, (100, 100, 100), -1)
    cv2.circle(img, (2*size//3, size//3), 8, (100, 100, 100), -1)
    
    # Nose
    pts = np.array([[size//2, size//2], [size//2-5, 2*size//3], [size//2+5, 2*size//3]], np.int32)
    cv2.fillPoly(img, [pts], (200, 180, 160))
    
    # Mouth
    cv2.ellipse(img, (size//2, 3*size//4), (size//6, size//12), 0, 0, 180, (180, 100, 100), 2)
    
    # Add subtle noise (natural texture)
    noise = np.random.randint(-10, 10, (size, size, 3))
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return img


def generate_fake_image(size=224):
    """
    Generate synthetic "fake" image with deepfake artifacts:
    - Blending boundaries
    - Color inconsistencies
    - Over-smoothing
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create less natural background with color shifts
    for i in range(size):
        for j in range(size):
            r = int(100 + 50 * np.random.rand())
            g = int(130 + 40 * np.random.rand())
            b = int(160 + 30 * np.random.rand())
            img[i, j] = [b, g, r]
    
    # Add face with visible blending boundary
    center = (size // 2, size // 2)
    axes = (size // 3, size // 2)
    
    # Inner face (different color - swapped)
    cv2.ellipse(img, center, axes, 0, 0, 360, (180, 220, 200), -1)
    
    # Obvious boundary artifact
    cv2.ellipse(img, center, axes, 0, 0, 360, (0, 255, 0), 3)
    
    # Features with wrong proportions (AI artifact)
    # Misaligned eyes
    cv2.circle(img, (size//3, size//3+5), 10, (80, 80, 80), -1)
    cv2.circle(img, (2*size//3, size//3-5), 8, (80, 80, 80), -1)
    
    # Asymmetric nose
    pts = np.array([[size//2+3, size//2], [size//2-8, 2*size//3], [size//2+2, 2*size//3]], np.int32)
    cv2.fillPoly(img, [pts], (170, 190, 170))
    
    # Unnatural mouth
    cv2.ellipse(img, (size//2+5, 3*size//4), (size//5, size//15), 0, 0, 180, (200, 80, 80), 3)
    
    # Over-smoothing effect (blur)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    
    # Add compression artifacts (block patterns)
    block_size = 8
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            if (i//block_size + j//block_size) % 2 == 0:
                img[i:i+block_size, j:j+block_size] = cv2.add(
                    img[i:i+block_size, j:j+block_size],
                    np.ones((min(block_size, size-i), min(block_size, size-j), 3), dtype=np.uint8) * 5
                )
    
    return img


def generate_dataset(output_dir, num_samples=1000, image_size=224):
    """Generate sample dataset"""
    
    output_path = Path(output_dir)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for cls in ['real', 'fake']:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Calculate splits (70/15/15)
    train_n = int(num_samples * 0.7)
    val_n = int(num_samples * 0.15)
    test_n = num_samples - train_n - val_n
    
    splits = {
        'train': train_n,
        'val': val_n,
        'test': test_n
    }
    
    print(f"Generating {num_samples} sample images...")
    print(f"  Train: {train_n} | Val: {val_n} | Test: {test_n}")
    
    for split, count in splits.items():
        print(f"\nGenerating {split} split...")
        
        # Generate real images
        for i in tqdm(range(count), desc=f'{split}/real'):
            img = generate_real_image(image_size)
            path = output_path / split / 'real' / f'real_{i:06d}.jpg'
            cv2.imwrite(str(path), img)
        
        # Generate fake images
        for i in tqdm(range(count), desc=f'{split}/fake'):
            img = generate_fake_image(image_size)
            path = output_path / split / 'fake' / f'fake_{i:06d}.jpg'
            cv2.imwrite(str(path), img)
    
    print("\n" + "=" * 70)
    print("Sample Dataset Generated!")
    print("=" * 70)
    print(f"\nLocation: {output_dir}")
    print(f"Total images: {num_samples * 2:,} ({num_samples:,} real + {num_samples:,} fake)")
    print("\nNext step: Start training")
    print(f"  python backend/training/train_efficientnet.py --data-dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate sample dataset for testing')
    parser.add_argument('--output', type=str, default='data/sample',
                        help='Output directory')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples per class')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Sample Dataset Generator")
    print("For testing the training pipeline")
    print("=" * 70)
    print(f"\n⚠️  NOTE: This generates synthetic data for TESTING ONLY")
    print(f"   For real deepfake detection, use FaceForensics++ or Celeb-DF")
    
    generate_dataset(args.output, args.num_samples, args.image_size)


if __name__ == '__main__':
    main()
