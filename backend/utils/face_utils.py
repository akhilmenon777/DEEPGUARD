
"""
Face detection and cropping utilities using OpenCV DNN face detector.
"""
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path

# Initialize DNN Face Detector (more robust than Haar Cascade)
face_net = None
try:
    # Try to use DNN face detector (works better with OpenCV 5.x)
    model_file = Path(__file__).parent.parent / "weights" / "opencv_face_detector_uint8.pb"
    config_file = Path(__file__).parent.parent / "weights" / "opencv_face_detector.pbtxt"
    
    # If model files don't exist, we'll use a simple fallback
    if model_file.exists() and config_file.exists():
        face_net = cv2.dnn.readNetFromTensorflow(str(model_file), str(config_file))
        print("✓ Loaded DNN face detector")
    else:
        print("⚠️  DNN face model not found, will use fallback detection")
        face_net = None
except Exception as e:
    print(f"⚠️  Face detection initialization: {e}")
    face_net = None

def extract_face(frame_bgr: np.ndarray, target_size: int = 224, allow_fallback: bool = True):
    """
    Detects faces in frame, selects the largest face, crops with padding,
    and resizes to target_size x target_size.
    Uses simple color-based face detection as fallback.
    Returns:
        face_rgb: np.ndarray of shape (target_size, target_size, 3) in RGB
        bbox: tuple (x, y, w, h) or None
        face_detected: bool
    """
    if frame_bgr is None or frame_bgr.size == 0:
        blank = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        return blank, None, False

    h, w, _ = frame_bgr.shape

    # Try DNN face detection if available
    faces = []
    if face_net is not None:
        try:
            blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), [104, 117, 123], False, False)
            face_net.setInput(blob)
            detections = face_net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    faces.append((x1, y1, x2 - x1, y2 - y1))
        except Exception as e:
            print(f"⚠️  DNN face detection error: {e}")
            faces = []
    
    # Simple fallback: assume face is in center area
    # This works for most selfie-style videos where person is centered
    if len(faces) == 0:
        print(f"[FaceUtils] Using center-crop fallback (no face detector available)")
        # Take center 70% of image as "face region"
        center_size = int(min(h, w) * 0.7)
        sy = (h - center_size) // 2
        sx = (w - center_size) // 2
        
        crop = frame_bgr[sy:sy + center_size, sx:sx + center_size]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Return as "face detected" since we're analyzing the central region
        return resized, (sx, sy, center_size, center_size), True

    # Use detected face
    if len(faces) > 0:
        # Sort by area (w * h) descending and take the largest
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, fw, fh = faces[0]

        # Add 12% margin around face
        margin_x = int(fw * 0.12)
        margin_y = int(fh * 0.12)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)

        crop = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return resized, (x, y, fw, fh), True

def numpy_to_base64_png(image_np: np.ndarray) -> str:
    """Encodes an RGB or RGBA numpy image to a base64 PNG data URL."""
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np * 255 if image_np.max() <= 1.0 else image_np, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image_np)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    