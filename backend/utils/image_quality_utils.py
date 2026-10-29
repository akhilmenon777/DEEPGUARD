"""
Image quality and blur detection utilities.
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Any


def detect_blur_laplacian(image_path: str) -> Dict[str, Any]:
    """
    Detects blur in an image using Laplacian variance method.
    Lower variance indicates more blur.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing blur score, verdict, and edge visualization
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Determine verdict based on threshold
    # Threshold: < 100 = Blurry, 100-200 = Moderate, > 200 = Sharp
    if laplacian_var < 100:
        verdict = "Blurry"
        quality = "Low"
        confidence = 100 - (laplacian_var / 100 * 100)
    elif laplacian_var < 200:
        verdict = "Moderate Quality"
        quality = "Medium"
        confidence = 50 + ((laplacian_var - 100) / 100 * 30)
    else:
        verdict = "Sharp"
        quality = "High"
        confidence = min(95, 80 + ((laplacian_var - 200) / 500 * 15))
    
    # Calculate additional metrics
    sharpness_score = min(100, laplacian_var / 5)
    
    # Edge detection for visualization
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Frequency domain analysis
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    high_freq_energy = np.sum(magnitude_spectrum[gray.shape[0]//3:, gray.shape[1]//3:])
    total_energy = np.sum(magnitude_spectrum)
    high_freq_ratio = (high_freq_energy / total_energy) * 100 if total_energy > 0 else 0
    
    return {
        "laplacian_variance": float(laplacian_var),
        "verdict": verdict,
        "quality": quality,
        "confidence": round(confidence, 1),
        "sharpness_score": round(sharpness_score, 1),
        "high_freq_ratio": round(high_freq_ratio, 2),
        "original_image": image,
        "edges_image": edges_colored,
        "image_shape": image.shape,
        "features_analyzed": [
            "Laplacian variance for edge sharpness",
            "Frequency domain high-frequency energy",
            "Canny edge detection visualization",
            "Overall image quality assessment"
        ]
    }


def extract_image_features(image_path: str) -> Dict[str, Any]:
    """
    Extracts various features from an image for quality assessment.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing extracted features
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Contrast (standard deviation)
    contrast = np.std(gray)
    
    # Brightness (mean intensity)
    brightness = np.mean(gray)
    
    # Sharpness (gradient magnitude)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = np.mean(np.sqrt(gx**2 + gy**2))
    
    # Noise estimate (using local standard deviation)
    kernel_size = 5
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
    variance = sqr_mean - mean**2
    noise_estimate = np.mean(np.sqrt(np.maximum(variance, 0)))
    
    return {
        "contrast": float(contrast),
        "brightness": float(brightness),
        "sharpness": float(sharpness),
        "noise_estimate": float(noise_estimate),
        "resolution": f"{image.shape[1]}x{image.shape[0]}"
    }
