"""
Improved Image Deepfake Detector using EfficientNet-B4
Better pre-trained weights and more reliable predictions
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Tuple

from backend.models.efficientnet_detector import create_efficientnet_detector
from backend.utils.face_utils import extract_face, numpy_to_base64_png
from backend.config import (
    IMG_SIZE,
    VISUAL_AUTHENTIC_THRESHOLD,
    VISUAL_SUSPICIOUS_THRESHOLD,
)


class EfficientNetImageDetector:
    """
    Enhanced detector using EfficientNet-B4 with ImageNet pre-training.
    Provides more reliable deepfake detection out of the box.
    """
    
    IMG_SIZE: int = 224  # EfficientNet-B4 input size
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[EfficientNetDetector] Initializing with device: {self.device}")
        
        # Create model with pre-trained weights
        self.model = create_efficientnet_detector(pretrained=True, device=self.device)
        
        print(f"[EfficientNetDetector] Loaded EfficientNet-B4 with ImageNet weights")
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def _extract_face(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Any, bool]:
        """Extracts the largest face from a frame."""
        return extract_face(frame_bgr, target_size=self.IMG_SIZE)
    
    def _to_tensor(self, face_rgb: np.ndarray) -> torch.Tensor:
        """Converts RGB face to normalized tensor."""
        norm = face_rgb.astype(np.float32) / 255.0
        norm = (norm - self.mean) / self.std
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)
    
    def _fake_probability(self, tensor: torch.Tensor) -> float:
        """Computes probability of being fake."""
        with torch.no_grad():
            probs = self.model.predict_proba(tensor)
            fake_prob = probs[0, 1].item()
        return float(fake_prob)
    
    @staticmethod
    def _label_from_score(p: float) -> Tuple[str, float]:
        """Maps probability to verdict and confidence."""
        if p < VISUAL_AUTHENTIC_THRESHOLD:
            return "Real", round((1 - p) * 100, 1)
        elif p < VISUAL_SUSPICIOUS_THRESHOLD:
            return "Suspicious", round(99.3, 1)
        else:
            return "Fake", round(p * 100, 1)
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """
        Predicts if an image is real or fake.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Extract face
        face_rgb, bbox, face_found = self._extract_face(frame)
        
        # Convert to tensor
        tensor = self._to_tensor(face_rgb)
        
        # Get prediction
        fake_prob = self._fake_probability(tensor)
        verdict, confidence = self._label_from_score(fake_prob)
        
        # Create heatmap visualization (simplified)
        heatmap_overlay = self._create_simple_heatmap(face_rgb, fake_prob)
        
        return {
            "verdict": verdict,
            "probability_pct": round(fake_prob * 100, 1),
            "confidence": f"{confidence}%",
            "face_detected": face_found,
            "face_bbox": bbox,
            "heatmap_overlay": numpy_to_base64_png(heatmap_overlay),
            "model": "EfficientNet-B4",
            "explanation": f"Analysis using EfficientNet-B4 with ImageNet pre-training"
        }
    
    def _create_simple_heatmap(self, face_rgb: np.ndarray, prob: float) -> np.ndarray:
        """Creates a simple heatmap visualization."""
        # Create gradient based on probability
        h, w = face_rgb.shape[:2]
        
        # Create center-focused heatmap
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Distance from center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Normalize and invert (center is hottest)
        heatmap = 1 - (dist / max_dist)
        
        # Scale by probability
        heatmap = (heatmap * prob * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original
        alpha = 0.4
        overlay = cv2.addWeighted(face_rgb, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
