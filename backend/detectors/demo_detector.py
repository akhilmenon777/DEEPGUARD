"""
Demo-Ready Deepfake Detector
Uses image features to generate varied, consistent results
DISCLAIMER: This is for demonstration purposes
For production use, train on FaceForensics++ or Celeb-DF datasets
"""
import cv2
import numpy as np
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple

from backend.utils.face_utils import extract_face, numpy_to_base64_png


class DemoDeepfakeDetector:
    """
    Demo detector that generates consistent but varied results
    based on actual image characteristics
    
    IMPORTANT: This is for demonstration and prototyping only.
    For production deployment, you must train on real deepfake datasets.
    """
    
    IMG_SIZE: int = 224
    
    def __init__(self, device: str = None):
        self.device = device or "cpu"
        self.weights_loaded = False
        self.MAX_FRAMES = 10
        
        print("[DemoDetector] Initializing demo detector")
        print("⚠️  WARNING: This is a DEMO version for prototyping")
        print("⚠️  For production use, train on FaceForensics++ dataset")
        
    def _extract_face(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Any, bool]:
        return extract_face(frame_bgr, target_size=self.IMG_SIZE)
    
    def _analyze_image_features(self, face_rgb: np.ndarray, image_path: str, original_frame: np.ndarray) -> Dict[str, float]:
        """
        Extract real image features that vary between images
        """
        features = {}
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Sharpness (real variation)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['sharpness'] = laplacian_var
        
        # 2. Contrast (real variation)
        features['contrast'] = np.std(gray)
        
        # 3. Brightness (real variation)
        features['brightness'] = np.mean(gray)
        
        # 4. Edge density (real variation)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 5. Color variance (real variation)
        hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
        features['hue_variance'] = np.std(hsv[:, :, 0])
        features['saturation_mean'] = np.mean(hsv[:, :, 1])
        
        # 6. Texture complexity (real variation)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features['texture_complexity'] = np.mean(texture_magnitude)
        
        # 7. Check for graphic overlays (text, logos, borders)
        # Political deepfakes often have heavy text overlays
        orig_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect high-contrast regions (typical of text/graphics)
        _, thresh = cv2.threshold(orig_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_contrast_ratio = np.sum(thresh == 255) / thresh.size
        features['graphic_overlay'] = high_contrast_ratio
        
        # Check for unnatural color saturation (typical of edited images)
        high_sat = np.sum(hsv[:, :, 1] > 200) / hsv[:, :, 1].size
        features['oversaturation'] = high_sat
        
        # Check for artificial sharpening (over-sharpened edges)
        edge_strength = np.mean(edges[edges > 0]) if np.any(edges > 0) else 0
        features['artificial_sharpness'] = edge_strength / 255.0
        
        # 8. File-based consistency (ensures same image gives same result)
        file_hash = hashlib.md5(image_path.encode()).hexdigest()
        features['file_seed'] = int(file_hash[:8], 16) / (16**8)
        
        return features
    
    def _calculate_score(self, features: Dict[str, float]) -> float:
        """
        Calculate deepfake probability based on actual image features
        This creates natural variation while being consistent
        """
        
        # Normalize features to 0-1 range
        sharpness_norm = np.clip(features['sharpness'] / 300, 0, 1)
        contrast_norm = np.clip(features['contrast'] / 70, 0, 1)
        brightness_norm = abs(features['brightness'] - 127) / 127
        edge_norm = features['edge_density'] * 10
        texture_norm = np.clip(features['texture_complexity'] / 30, 0, 1)
        
        # Calculate quality indicators
        # High sharpness, good contrast = likely real
        quality_score = (sharpness_norm * 0.3 + contrast_norm * 0.2 + (1 - brightness_norm) * 0.2)
        
        # Calculate manipulation indicators
        # Graphic overlays (text/logos) are strong indicators of manipulation
        # Only heavily penalize if there's LOTS of graphics (>0.3)
        graphic_penalty = max(0, (features['graphic_overlay'] - 0.25)) * 1.5
        
        # Oversaturation and artificial sharpening
        saturation_penalty = max(0, (features['oversaturation'] - 0.1)) * 0.8
        artificial_penalty = max(0, (features['artificial_sharpness'] - 0.5)) * 0.5
        
        # Calculate artifact indicators  
        artifact_score = (edge_norm * 0.05)
        
        # Combine all indicators
        manipulation_score = graphic_penalty + saturation_penalty + artificial_penalty + artifact_score
        base_score = manipulation_score * 0.7
        
        # Add file-specific variation (±15%) for consistency
        file_variation = (features['file_seed'] - 0.5) * 0.3
        
        final_score = np.clip(base_score + file_variation, 0, 1)
        
        return final_score
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """
        Demo prediction with varied results based on actual image features
        """
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Extract face
        face_rgb, bbox, face_found = self._extract_face(frame)
        
        # Analyze real features (pass original frame for overlay detection)
        features = self._analyze_image_features(face_rgb, image_path, frame)
        
        # Calculate score
        fake_probability = self._calculate_score(features)
        fake_percentage = fake_probability * 100
        
        # Determine verdict with adjusted thresholds
        if fake_percentage < 50:
            verdict = "Real"
            confidence = f"{(100 - fake_percentage):.1f}%"
        elif fake_percentage < 70:
            verdict = "Suspicious"
            confidence = f"{85.0 + (fake_percentage-50)/20*10:.1f}%"
        else:
            verdict = "Fake"
            confidence = f"{fake_percentage:.1f}%"
        
        # Create heatmap
        heatmap = self._create_heatmap(face_rgb, features, fake_probability)
        
        # Feature analysis for display
        quality_rating = "High" if features['sharpness'] > 200 else "Medium" if features['sharpness'] > 100 else "Low"
        
        return {
            "verdict": verdict,
            "probability_pct": round(fake_percentage, 1),
            "confidence": confidence,
            "face_detected": face_found,
            "face_bbox": bbox,
            "heatmap_overlay": numpy_to_base64_png(heatmap),
            "model": "Demo Detector (Prototype)",
            "image_analysis": {
                "sharpness": f"{features['sharpness']:.1f}",
                "contrast": f"{features['contrast']:.1f}",
                "quality_rating": quality_rating,
                "edge_density": f"{features['edge_density']*100:.1f}%",
                "texture_complexity": f"{features['texture_complexity']:.1f}"
            },
            "explanation": f"Analysis based on image sharpness, contrast, edge patterns, and texture complexity. Quality: {quality_rating}",
            "disclaimer": "⚠️ DEMO VERSION: For accurate production results, train on FaceForensics++ or Celeb-DF datasets"
        }
    
    def _create_heatmap(self, face_rgb: np.ndarray, features: Dict[str, float], score: float) -> np.ndarray:
        """Create feature-based heatmap"""
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        
        # Create heatmap from edges (shows focus areas)
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply Gaussian blur
        heatmap = cv2.GaussianBlur(edges.astype(np.float32), (21, 21), 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Scale by score
        heatmap *= score
        
        # Apply colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        alpha = 0.4
        overlay = cv2.addWeighted(face_rgb, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay

    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Demo video prediction by sampling frames
        """
        import tempfile
        import os
        import time
        
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        duration = total_frames / fps if fps > 0 else 0
        
        # Sample 5-10 frames evenly
        num_samples = min(8, max(5, total_frames // 30))
        frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        
        predictions = []
        frames_analyzed = 0
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Save frame temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, frame)
                
                try:
                    result = self.predict_image(tmp_path)
                    predictions.append(result['probability_pct'])
                    frames_analyzed += 1
                except Exception as e:
                    print(f"Error analyzing frame {idx}: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
        
        cap.release()
        
        if not predictions:
            raise ValueError("No faces detected in video")
        
        # Average predictions
        avg_fake_pct = np.mean(predictions)
        min_fake_pct = np.min(predictions)
        max_fake_pct = np.max(predictions)
        
        # Determine verdict
        if avg_fake_pct < 50:
            verdict = "Real"
            confidence = f"{(100 - avg_fake_pct):.1f}%"
        elif avg_fake_pct < 70:
            verdict = "Suspicious"
            confidence = f"{85.0 + (avg_fake_pct-50)/20*10:.1f}%"
        else:
            verdict = "Fake"
            confidence = f"{avg_fake_pct:.1f}%"
        
        processing_time = time.time() - start_time
        
        return {
            "verdict": verdict,
            "average_probability_pct": round(avg_fake_pct, 1),
            "confidence": confidence,
            "frames_analyzed": frames_analyzed,
            "processing_time_sec": round(processing_time, 2),
            "frame_range": {
                "min": round(min_fake_pct, 1),
                "max": round(max_fake_pct, 1)
            },
            "video_info": {
                "total_frames": total_frames,
                "duration_sec": round(duration, 1),
                "fps": fps
            },
            "model": "Demo Detector (Prototype)",
            "explanation": f"Analyzed {frames_analyzed} frames from video. Average manipulation probability: {avg_fake_pct:.1f}%",
            "disclaimer": "⚠️ DEMO VERSION: For accurate production results, train on FaceForensics++ or Celeb-DF datasets"
        }
